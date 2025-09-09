import torch

def masked_dwell_bonus(
    env,
    mask_name: str = "dof_mask",
    eps: float = 0.04,          # допуск по позе (норм. [-1,1])
    vel_eps: float = 0.03,      # допуск по скорости (норм.)
    hold_steps: int = 8,        # после скольки подряд «ОК» шагов начать платить бонус
    bonus: float = 0.5,         # базовый бонус (когда выдержали hold_steps)
    growth: float = 0.1,        # добавка за каждый следующий «ОК» шаг
    max_bonus: float = 3.0,     # верхняя планка
) -> torch.Tensor:
    """
    Платит всё больше, чем дольше ВСЕ активные DOF (mask==1) удерживаются в допуске
    по позиции и скорости. Счётчик сбрасывается, как только вышли из окна «ОК».
    """
    robot = env.scene["robot"]
    q, qd = robot.data.joint_pos, robot.data.joint_vel
    device, dtype = q.device, q.dtype

    # нормировка в [-1,1]
    qmin = robot.data.soft_joint_pos_limits[..., 0]
    qmax = robot.data.soft_joint_pos_limits[..., 1]
    mid  = 0.5 * (qmin + qmax)
    rng  = (qmax - qmin).clamp_min(1e-6)
    qn, qdn = 2.0 * (q - mid) / rng, 2.0 * qd / rng

    # цель и маска
    tgt = torch.as_tensor(
        env.command_manager.get_term("target_joint_pose").command,
        dtype=dtype, device=device
    )
    msk = (env.command_manager.get_term(mask_name).command > 0.5)

    # «ОК» одновременно по позе и скорости (только по активным DOF)
    ok_pos = (qn - tgt).abs() <= eps
    ok_vel = qdn.abs() <= vel_eps
    ok_all_j = torch.where(msk, ok_pos & ok_vel, torch.ones_like(ok_pos, dtype=torch.bool))
    ok_all = ok_all_j.all(dim=1)  # (N,)

    # держим счётчик по env
    if not hasattr(env, "_dwell_cnt"):
        env._dwell_cnt = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    env._dwell_cnt = torch.where(ok_all, env._dwell_cnt + 1, torch.zeros_like(env._dwell_cnt))

    # обнуляем счётчики на ресетах
    resets = env.termination_manager.terminated | env.termination_manager.time_outs
    if resets.any():
        env._dwell_cnt[resets] = 0

    # бонус: начинается после hold_steps и растёт на growth каждый шаг, ограничен max_bonus
    extra = (env._dwell_cnt - hold_steps).clamp_min(0).to(q.dtype)
    r = torch.clamp(bonus + growth * extra, max=max_bonus)
    r = torch.where(env._dwell_cnt >= hold_steps, r, torch.zeros_like(r))
    return r
