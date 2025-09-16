import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

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
