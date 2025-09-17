import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

# ---------- TARGET ----------

@torch.no_grad()
def respawn_reached_targets(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None,
                            reach_radius: float = 0.35, r_min: float = 2.0, r_max: float = 6.0, z: float = 0.05):
    robot  = env.scene["robot"]
    target = env.scene["target"]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long).view(-1)

    # считаем только на указанном подмножестве
    d = target.data.root_pos_w[env_ids, :2] - robot.data.root_pos_w[env_ids, :2]
    dist = torch.linalg.norm(d, dim=1)
    hit_local = torch.nonzero(dist < reach_radius, as_tuple=False).squeeze(-1)
    if hit_local.numel() > 0:
        hit_global = env_ids[hit_local]
        respawn_target(env, hit_global, r_min=r_min, r_max=r_max, z=z)


def respawn_target(env: ManagerBasedRLEnv, env_ids: torch.Tensor, r_min=2.0, r_max=6.0, z=0.05):
    device = env.device
    env_ids = env_ids.to(device=device, dtype=torch.long).view(-1)
    N = env_ids.numel()
    if N == 0:
        return

    robot  = env.scene["robot"]
    target = env.scene["target"]

    base_xy = robot.data.root_pos_w[env_ids, :2]  # (N,2)

    yaw = torch.rand(N, device=device) * (2 * torch.pi)
    rad = r_min + torch.rand(N, device=device) * (r_max - r_min)
    goal_xy = base_xy + torch.stack((rad * torch.cos(yaw), rad * torch.sin(yaw)), dim=1)

    pos  = target.data.root_pos_w   # (E,3)
    quat = target.data.root_quat_w  # (E,4)

    pos[env_ids, 0:2] = goal_xy
    pos[env_ids, 2]   = z

    pose7_env = torch.cat((pos[env_ids], quat[env_ids]), dim=1)  # (N,7)
    target.write_root_pose_to_sim(pose7_env, env_ids=env_ids)

# ---------- OBSTACLES (RESET ONLY) ----------

def _collect_obstacle_objects(env: ManagerBasedRLEnv):
    """Собирает список объектов препятствий obst_XX. Fallback: scene['obstacles'] (один объект)."""
    obst_objs = []
    if hasattr(env.scene, "rigid_objects") and isinstance(env.scene.rigid_objects, dict):
        names = [k for k in env.scene.rigid_objects.keys() if k.startswith("obst_")]
        obst_objs = [env.scene[k] for k in names]
    else:
        # по атрибутам
        names = [n for n in dir(env.scene) if n.startswith("obst_")]
        obst_objs = [getattr(env.scene, n) for n in names]

    if len(obst_objs) == 0 and "obstacles" in getattr(env.scene, "__getitem__", lambda k: {}) and isinstance(env.scene["obstacles"], object):
        # очень осторожный fallback на один объект
        return [env.scene["obstacles"]]
    return obst_objs


@torch.no_grad()
def spawn_obstacles_at_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    num_obstacles: tuple[int, int] = (6, 10),
    r_max: float = 6.5,
    keepout_robot: float = 0.8,
    keepout_goal: float = 0.8,
    min_obstacle_gap: float = 0.5,
    obstacle_z: float = 0.3,
) -> None:
    device = env.device
    N = env_ids.numel()

    robot  = env.scene["robot"]
    target = env.scene["target"]

    base_xy = robot.data.root_pos_w[env_ids, :2]   # (N,2)
    goal_xy = target.data.root_pos_w[env_ids, :2]  # (N,2) 

    coll = env.scene["obstacles"]  # RigidObjectCollection

    pos  = coll.data.object_pos_w   # (E, M, 3)
    quat = coll.data.object_quat_w  # (E, M, 4)

    if pos.shape[0] != env.scene.num_envs and pos.shape[1] == env.scene.num_envs:
        pos  = pos.permute(1, 0, 2)   # view
        quat = quat.permute(1, 0, 2)  # view

    E, M = pos.shape[0], pos.shape[1]
    if M == 0:
        return

    n_low, n_high = num_obstacles
    k_active = torch.randint(low=n_low, high=n_high + 1, size=(N,), device=device).clamp_(max=M)

    for i, e in enumerate(env_ids.tolist()):
        placed_xy = []
        attempts, need = 0, int(k_active[i].item())

        while len(placed_xy) < need and attempts < 300:
            attempts += 1
            x = (torch.rand((), device=device) - 0.5) * 2.0 * (r_max + 1.0)
            y = (torch.rand((), device=device) - 0.5) * 2.0 * (r_max + 1.0)
            p_xy = torch.stack((x, y)) + base_xy[i]

            ok_robot = (p_xy - base_xy[i]).norm() > keepout_robot
            ok_goal  = (p_xy - goal_xy[i]).norm()  > keepout_goal
            ok_gap   = True if not placed_xy else torch.all(
                (torch.stack(placed_xy) - p_xy).norm(dim=1) > min_obstacle_gap
            )
            if ok_robot and ok_goal and ok_gap:
                placed_xy.append(p_xy)

        for j in range(M):
            if j < len(placed_xy):
                pos[e, j, 0:2] = placed_xy[j]
                pos[e, j, 2]   = obstacle_z
            else:
                pos[e, j, 2]   = -1.0

    pose7 = torch.cat((pos, quat), dim=-1)
    coll.write_object_pose_to_sim(pose7)
    
def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Берём yaw (рад) из кватерниона(ов) формата (w,x,y,z)."""
    # euler_xyz_from_quat возвращает (roll, pitch, yaw)
    _, _, yaw = euler_xyz_from_quat(q, wrap_to_2pi=False)
    return wrap_to_pi(yaw)


@torch.no_grad()
def commands_towards_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,       
    command_name: str = "base_velocity",
    lin_speed: float = 0.6,
    max_yaw_rate: float = 2.0,
    stop_radius: float = 0.35,
    slow_radius: float = 1.2,
    yaw_kp: float = 2.0,
    holonomic: bool = True,
) -> torch.Tensor:
    # нормализуем env_ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long).view(-1)
    if env_ids.numel() == 0:
        return torch.zeros((0, 3), device=env.device)

    robot  = env.scene["robot"]
    target = env.scene["target"]

    # позиции только для выбранных env
    pr = robot.data.root_pos_w[env_ids, :2]    # (N,2)
    pt = target.data.root_pos_w[env_ids, :2]   # (N,2)
    yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w[env_ids])  # (N,)

    d = pt - pr
    dist = torch.linalg.norm(d, dim=1)
    heading = torch.atan2(d[:, 1], d[:, 0])

    tiny = dist < 1e-9
    if tiny.any():
        heading = torch.where(tiny, yaw, heading)

    err_yaw = wrap_to_pi(heading - yaw)

    # профиль скорости
    denom = float(max(slow_radius - stop_radius, 1e-6))
    speed_scale = ((dist - stop_radius) / denom).clamp(0.0, 1.0)
    v = lin_speed * speed_scale

    if holonomic:
        vx = v * torch.cos(err_yaw)
        vy = v * torch.sin(err_yaw)
    else:
        vx = v * torch.cos(err_yaw)
        vy = torch.zeros_like(vx)

    wz = (yaw_kp * err_yaw).clamp(-max_yaw_rate, max_yaw_rate)

    near = dist < stop_radius
    if near.any():
        z = torch.zeros_like(v)
        vx = torch.where(near, z, vx)
        vy = torch.where(near, z, vy)
        wz = torch.where(near, z, wz)

    cmd = torch.stack((vx, vy, wz), dim=1)  # (N,3)

    # запись в command_manager только по env_ids
    cm = getattr(env, "command_manager", None)
    if cm is not None:
        term = cm.get_term(command_name)
        cmd = cmd.to(device=term.command.device, dtype=term.command.dtype)

        N_dst, C_dst = term.command.shape
        _, C_src = cmd.shape
        k = min(C_src, C_dst)

        if N_dst == env.scene.num_envs:
            term.command[env_ids, :k].copy_(cmd[:, :k])
        elif N_dst == 1:
            # буфер «общий на все» — кладём первую строку
            term.command[:1, :k].copy_(cmd[:1, :k])
        else:
            # редкий случай несоответствия формы
            raise RuntimeError(
                f"Command buffer shape {term.command.shape} "
                f"не согласуется с env_ids (N={env.scene.num_envs})."
            )

    return cmd
