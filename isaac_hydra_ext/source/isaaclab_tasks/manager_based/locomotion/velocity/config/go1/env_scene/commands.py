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
    env,
    command_name: str = "base_velocity",
    lin_speed: float = 0.6,      # м/с — максимальная поступательная скорость
    max_yaw_rate: float = 2.0,   # рад/с — ограничение рысканья
    stop_radius: float = 0.35,   # радиус полной остановки
    slow_radius: float = 1.2,    # начиная отсюда плавно замедляемся
    yaw_kp: float = 2.0,         # П-коэффициент на ошибку курса
    holonomic: bool = True,      # если False — vy=0
) -> torch.Tensor:
    """
    Сгенерировать и записать команды [vx, vy, wz] для КАЖДОГО env:
      - курс на свой target, скорость линейно падает от lin_speed до 0 в [stop_radius..slow_radius]
      - внутри stop_radius полная остановка
      - команды заданы в СК робота
    Возвращает тензор команд формы (N, 3).
    """
    # --- доступ к актёрам сцены
    robot = env.scene["robot"]
    try:
        target = env.scene["target"]
    except KeyError as e:
        raise KeyError(
            "Scene object 'target' not found. Убедитесь, что цель добавлена как per-env "
            "с путём вида prim_path='{ENV_REGEX_NS}/Target'."
        ) from e

    # --- позиции/курсы
    pr = robot.data.root_pos_w[:, :2]                 # (N,2)
    pt_all = target.data.root_pos_w                   # ожидаем (N,3)
    # строгая проверка батча — чтобы не было тихого бродкаста одной цели на все env
    if pt_all.shape[0] != pr.shape[0]:
        raise RuntimeError(
            f"'target' batch mismatch: targets={pt_all.shape[0]} vs robots={pr.shape[0]}. "
            "Проверьте, что target размножён по env (prim_path='{ENV_REGEX_NS}/Target')."
        )
    pt = pt_all[:, :2]                                # (N,2)

    # yaw (рад) из кватерниона WXYZ мира → поворот вокруг +Z
    yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w) # (N,)

    # --- геометрия до цели
    d = pt - pr                                       # (N,2)
    dist = torch.linalg.norm(d, dim=1)                # (N,)
    heading = torch.atan2(d[:, 1], d[:, 0])           # (N,)

    # если очень близко к цели — чтобы не шумел atan2, считаем err_yaw=0
    tiny = dist < 1e-9
    if tiny.any():
        heading = torch.where(tiny, yaw, heading)

    # ошибка курса
    err_yaw = wrap_to_pi(heading - yaw)               # (N,)

    # --- профиль скорости: линейно в диапазоне [stop_radius .. slow_radius]
    denom = float(max(slow_radius - stop_radius, 1e-6))
    speed_scale = ((dist - stop_radius) / denom).clamp(0.0, 1.0)  # (N,)
    v = lin_speed * speed_scale                                   # (N,)

    # --- команды в СК робота
    if holonomic:
        vx = v * torch.cos(err_yaw)
        vy = v * torch.sin(err_yaw)
    else:
        # для не-холономной базы нет боковой скорости
        vx = v * torch.cos(err_yaw)  # можно поставить просто vx = v, если так принято в вашем контроллере
        vy = torch.zeros_like(vx)

    wz = (yaw_kp * err_yaw).clamp(-max_yaw_rate, max_yaw_rate)

    # полная остановка рядом с целью
    near = dist < stop_radius
    if near.any():
        z = torch.zeros_like(v)
        vx = torch.where(near, z, vx)
        vy = torch.where(near, z, vy)
        wz = torch.where(near, z, wz)

    cmd = torch.stack((vx, vy, wz), dim=1)            # (N,3)

    # --- запись в command_manager с аккуратным согласованием форм/типов
    cm = getattr(env, "command_manager", None)
    if cm is not None:
        term = cm.get_term(command_name)

        # приведение типов/девайса к приёмнику
        cmd = cmd.to(device=term.command.device, dtype=term.command.dtype)

        N_src, C_src = cmd.shape
        N_dst, C_dst = term.command.shape
        k = min(C_src, C_dst)

        if N_dst == N_src:
            term.command[:, :k].copy_(cmd[:, :k])
        elif N_dst == 1:
            # буфер приёмника один на все — положим первую строку
            term.command[:1, :k].copy_(cmd[:1, :k])
        elif N_src == 1:
            # на источник пришёл один вектор — растиражируем
            term.command[:, :k].copy_(cmd.expand(N_dst, -1)[:, :k])
        else:
            raise RuntimeError(
                f"Batch mismatch on write: cmd {(N_src, C_src)} -> buffer {(N_dst, C_dst)}. "
                f"Приведите один из батчей к 1 или к одинаковому N."
            )

    return cmd


