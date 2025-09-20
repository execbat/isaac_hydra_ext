import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

@configclass
class ChaseCurriculumCfg:
    # --- Phase A: Acceleration and balance (0 → 200k steps) ---
    # We immediately provide strong tracking, soft orientation, without penalties.
    # (track_lin_vel_xy_exp уже = 2.0 at start already)
    upr_0 = CurrTerm(func=mdp.modify_reward_weight,
                     params={"term_name": "upright", "weight": 0.5, "num_steps": 0})
    head_0 = CurrTerm(func=mdp.modify_reward_weight,
                      params={"term_name": "heading_align", "weight": 0.25, "num_steps": 0})

    # --- Phase B: Stabilization and Step Pattern (turning on the “air of time”) ---
    air_on_80k = CurrTerm(func=mdp.modify_reward_weight,
                          params={"term_name": "feet_air_time", "weight": 0.05, "num_steps": 80_000})
    air_more_250k = CurrTerm(func=mdp.modify_reward_weight,
                             params={"term_name": "feet_air_time", "weight": 0.10, "num_steps": 250_000})

    # --- Phase C: Tightening up the accuracy ---
    # We gradually increase the smoothing of actions and the penalty for the moment
    act_150k = CurrTerm(func=mdp.modify_reward_weight,
                        params={"term_name": "action_rate_l2", "weight": -0.005, "num_steps": 150_000})
    act_300k = CurrTerm(func=mdp.modify_reward_weight,
                        params={"term_name": "action_rate_l2", "weight": -0.010, "num_steps": 300_000})

    tq_150k = CurrTerm(func=mdp.modify_reward_weight,
                       params={"term_name": "dof_torques_l2", "weight": -1e-5, "num_steps": 150_000})
    tq_350k = CurrTerm(func=mdp.modify_reward_weight,
                       params={"term_name": "dof_torques_l2", "weight": -2e-5, "num_steps": 350_000})

    # bumps and slips - only after mastering the basic gait
    imp_200k = CurrTerm(func=mdp.modify_reward_weight,
                        params={"term_name": "feet_impact_vel", "weight": -0.02, "num_steps": 200_000})
    imp_400k = CurrTerm(func=mdp.modify_reward_weight,
                        params={"term_name": "feet_impact_vel", "weight": -0.05, "num_steps": 400_000})

    slip_300k = CurrTerm(func=mdp.modify_reward_weight,
                         params={"term_name": "feet_slide", "weight": -0.05, "num_steps": 300_000})

    # --- Phase D: Increasing the tracking requirements once it's "beautiful" ---
    tr_lin_200k = CurrTerm(func=mdp.modify_reward_weight,
                           params={"term_name": "track_lin_vel_xy_exp", "weight": 3.0, "num_steps": 200_000})
    tr_lin_400k = CurrTerm(func=mdp.modify_reward_weight,
                           params={"term_name": "track_lin_vel_xy_exp", "weight": 4.0, "num_steps": 400_000})
    tr_yaw_200k = CurrTerm(func=mdp.modify_reward_weight,
                           params={"term_name": "track_ang_vel_z_exp", "weight": 1.5, "num_steps": 200_000})
    tr_yaw_400k = CurrTerm(func=mdp.modify_reward_weight,
                           params={"term_name": "track_ang_vel_z_exp", "weight": 2.0, "num_steps": 400_000})

    # --- (optional) command complexity scheduling: small → full ---
    # cmd_easy_0   = CurrTerm(func=mdp.modify_env_param,
    #   params={"address": "commands.base_velocity.ranges.lin_vel_x", "value": (-0.5, 0.5), "num_steps": 0})
    # cmd_full_300k = CurrTerm(func=mdp.modify_env_param,
    #   params={"address": "commands.base_velocity.ranges.lin_vel_x", "value": (-1.5, 1.5), "num_steps": 300_000})
