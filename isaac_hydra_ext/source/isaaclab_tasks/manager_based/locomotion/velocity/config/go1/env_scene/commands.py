from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.utils.math import quat_to_yaw, wrap_to_pi

@torch.no_grad()
def commands_towards_target(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    lin_speed: float = 0.6,            # м/с
    max_yaw_rate: float = 2.0,         # рад/с
    stop_radius: float = 0.35,         # остановка у цели
    slow_radius: float = 1.2,          # плавное снижение скорости
) -> torch.Tensor:
    """Генерирует команды [vx, vy, wz] к marker-цели и кладёт их в command_manager."""
    robot = env.scene["robot"]
    tgt   = env.scene["target"]

    # position in the worls
    p_r = robot.data.root_pos_w[:, :2]            # [N,2]
    yaw = quat_to_yaw(robot.data.root_quat_w)     # [N]
    p_t = tgt.data.root_pos_w[:, :2]              # [N,2]

    d   = p_t - p_r
    dist = torch.linalg.norm(d, dim=1)            # [N]
    heading = torch.atan2(d[:, 1], d[:, 0])       # [N]
    err_yaw = wrap_to_pi(heading - yaw)           # [N]

    # desired linear speed (slow down near the target)
    speed_scale = torch.clamp((dist - stop_radius) / (slow_radius - stop_radius + 1e-6), 0.0, 1.0)
    v = lin_speed * speed_scale                   # [N]

    # компоненты в корпусной СК: vx вдоль курса на цель, vy — поперёк
    vx = v * torch.cos(err_yaw)
    vy = v * torch.sin(err_yaw)
    wz = torch.clamp(2.0 * err_yaw, -max_yaw_rate, max_yaw_rate)

    # set zeros if too close
    near = dist < stop_radius
    vx = torch.where(near, torch.zeros_like(vx), vx)
    vy = torch.where(near, torch.zeros_like(vy), vy)
    wz = torch.where(near, torch.zeros_like(wz), wz)

    cmd = torch.stack((vx, vy, wz), dim=1)  # [N,3]

    # синхронизируем с command_manager, чтобы реварды читали то же самое
    term = env.command_manager.get_term(command_name)
    term.set_command(cmd)

    return cmd

