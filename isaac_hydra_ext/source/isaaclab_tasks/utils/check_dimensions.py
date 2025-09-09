def _per_env_dim_from_space(space, num_envs: int | None = None) -> int:
    """Return dims of action/observation space of a single env"""
    if isinstance(space, spaces.Box):
        shape = space.shape
        if shape is None:
            shape = np.asarray(space.sample()).shape

        if num_envs is not None and len(shape) >= 1 and shape[0] == num_envs:
            shape = shape[1:]
        if len(shape) == 0:
            return 1
        return int(np.prod(shape))

    if isinstance(space, spaces.Discrete):
        return 1

    if isinstance(space, spaces.MultiBinary):
        return int(space.n)

    if isinstance(space, spaces.MultiDiscrete):
        return int(np.sum(space.nvec))

    if isinstance(space, spaces.Dict):
        # суммируем пер-энв размерности по ключам (обычно берём только 'policy', но общее решение — сумма)
        return int(sum(_per_env_dim_from_space(s, num_envs) for s in space.spaces.values()))

    if isinstance(space, spaces.Tuple):
        return int(sum(_per_env_dim_from_space(s, num_envs) for s in space.spaces))

    raise ValueError(f"Unsupported space type: {type(space)}")



