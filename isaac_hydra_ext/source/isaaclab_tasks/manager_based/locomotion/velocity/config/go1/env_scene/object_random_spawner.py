def sample_target_and_obstacles(env, env_ids: torch.Tensor,
                                r_min=2.0, r_max=6.0,
                                num_obstacles=(10, 10),
                                keepout_robot=0.8,
                                keepout_goal=0.8):
    """sample the sarget [r_min,r_max] and putting obstackes without collosions"""
    device = env.device
    N = len(env_ids)

    # --- target
    yaw0 = torch.rand(N, device=device) * 2*torch.pi
    rad  = r_min + torch.rand(N, device=device) * (r_max - r_min)
    dx = rad * torch.cos(yaw0)
    dy = rad * torch.sin(yaw0)
    base = env.scene["robot"].data.root_pos_w[env_ids, :2]
    goal_xy = base + torch.stack((dx, dy), dim=1)                   # [n,2]

    tgt = env.scene["target"]
    tgt_pos = tgt.data.root_pos_w.clone()
    tgt_pos[env_ids, 0:2] = goal_xy
    tgt_pos[env_ids, 2]   = 0.05                                    # чуть выше пола
    tgt.write_root_pose(positions=tgt_pos)

    # --- obstacles
    # сколько спавнить
    n_low, n_high = num_obstacles
    k = torch.randint(low=n_low, high=n_high+1, size=(N,), device=device)

    # array of obstacle poses
    obst = env.scene["obstacles"]
    pos  = obst.data.root_pos_w.clone()

    def _valid(p_xy, all_xy, min_dist):
        if all_xy.shape[0] == 0:
            return True
        return torch.all(torch.linalg.norm(all_xy - p_xy, dim=1) > min_dist)

    # attempt to put obstacles without collosions
    for i, e in enumerate(env_ids.tolist()):
        placed = []
        attempts = 0
        while len(placed) < int(k[i]) and attempts < 200:
            attempts += 1
            x = (torch.rand(1, device=device) - 0.5) * 2 * (r_max + 1.0)
            y = (torch.rand(1, device=device) - 0.5) * 2 * (r_max + 1.0)
            p = torch.stack((x.squeeze(), y.squeeze()))
            # move into robot CS
            p_world = env.scene["robot"].data.root_pos_w[e, :2] + p

            if (torch.linalg.norm(p_world - base[i]) > keepout_robot and
                torch.linalg.norm(p_world - goal_xy[i]) > keepout_goal and
                _valid(p_world, torch.stack(placed) if placed else torch.empty((0,2), device=device), 0.5)):
                placed.append(p_world)

        # record
        for j, xy in enumerate(placed):
            pos[e, 0] = xy[0]
            pos[e, 1] = xy[1]
            pos[e, 2] = 0.3  # half of the obstacles height

    obst.write_root_pose(positions=pos)

