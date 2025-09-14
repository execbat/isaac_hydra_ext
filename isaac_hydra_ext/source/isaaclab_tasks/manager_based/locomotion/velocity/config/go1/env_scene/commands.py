import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi


def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Берём yaw (рад) из кватерниона(ов) формата (w,x,y,z)."""
    # euler_xyz_from_quat возвращает (roll, pitch, yaw)
    _, _, yaw = euler_xyz_from_quat(q, wrap_to_2pi=False)
    return wrap_to_pi(yaw)


@torch.no_grad()
def commands_towards_target(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    lin_speed: float = 0.6,     # м/с (макс скорость поступательная)
    max_yaw_rate: float = 2.0,  # рад/с (сатурация по рысканью)
    stop_radius: float = 0.35,  # радиус полной остановки
    slow_radius: float = 1.2,   # откуда начинаем плавно замедляться
) -> torch.Tensor:
    """
    Генерирует команды [vx, vy, wz], направляющие робота к целевому маркеру,
    и записывает их в command_manager под ключом `command_name`.
    """
    robot = env.scene["robot"]
    try:
        target = env.scene["target"]
    except KeyError as e:
        raise KeyError("Scene object 'target' not found. Добавь объект цели в сцену с именем 'target'.") from e

    # Позиции в мире
    pr = robot.data.root_pos_w[:, :2]          # (N,2)
    pt = target.data.root_pos_w[:, :2]         # (N,2)
    yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w)  # (N,)

    # target vector and error
    d = pt - pr                                 # (N,2)
    dist = torch.linalg.norm(d, dim=1)          # (N,)
    heading = torch.atan2(d[:, 1], d[:, 0])     # (N,)
    err_yaw = wrap_to_pi(heading - yaw)         # (N,)

    # speed profile
    denom = max(slow_radius - stop_radius, 1e-6)
    speed_scale = ((dist - stop_radius) / denom).clamp(0.0, 1.0)  # (N,)
    v = lin_speed * speed_scale                                    # (N,)

    # speed in robot CS: vx вдоль цели, vy поперёк
    vx = v * torch.cos(err_yaw)
    vy = v * torch.sin(err_yaw)

    wz = (2.0 * err_yaw).clamp(-max_yaw_rate, max_yaw_rate)

    # full stop near of the target
    near = dist < stop_radius
    if near.any():
        z = torch.zeros_like(vx)
        vx = torch.where(near, z, vx)
        vy = torch.where(near, z, vy)
        wz = torch.where(near, z, wz)

    cmd = torch.stack((vx, vy, wz), dim=1)  # (N,3)

    cm = getattr(env, "command_manager", None)
    if cm is not None:
        term = cm.get_term(command_name)

        cmd = cmd.to(device=term.command.device, dtype=term.command.dtype)

        if term.command.shape == cmd.shape:
            term.command.copy_(cmd)
        else:
            k = min(term.command.shape[1], cmd.shape[1])
            term.command[:, :k].copy_(cmd[:, :k])
   
    return cmd

