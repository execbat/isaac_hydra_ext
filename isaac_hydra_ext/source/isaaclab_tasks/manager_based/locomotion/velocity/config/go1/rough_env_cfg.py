# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaac_hydra_ext.source.isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort: skip


@configclass
class UnitreeGo1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
#        self.decimation = 4 
#        self.sim.dt =  0.005
#        self.sim.render_interval = self.decimation
#        self.sim.render_interval = self.decimation
#        self.sim.physics_material = self.scene.terrain.physics_material
#        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
#        # update sensor update periods
#        # we tick all the sensors based on the smallest update period (physics update period)
#        if self.scene.height_scanner is not None:
#            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
#        if self.scene.contact_forces is not None:
#            self.scene.contact_forces.update_period = self.sim.dt
        
        
        
        
        self.episode_length_s = 40.0
        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.5

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.params["threshold"] = 0.30 
        self.rewards.feet_air_time.weight = 0.08     
        

        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -5e-6 
        self.rewards.track_lin_vel_xy_exp.weight = 12.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.dof_acc_l2.weight = -1e-7
        
        # penalties
        self.rewards.action_rate_l2.weight = -0.03
        
        self.rewards.flat_orientation_l2.weight = -2.0    
        self.rewards.dof_pos_limits.weight      = -0.1   
        
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.ang_vel_xy_l2.weight = -0.3
        self.rewards.track_lin_vel_xy_mse.weight = -3.0 # penalty for not following desired direction
        self.rewards.track_ang_vel_z_mse.weight = -1.5 # penalty for not following desired direction

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"
        #self.rewards.termination_penalty.weight = 0.0

@configclass
class UnitreeGo1RoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
#        self.decimation = 4 
#        self.sim.dt =  0.005
#        self.sim.render_interval = self.decimation
#        self.sim.render_interval = self.decimation
#        self.sim.physics_material = self.scene.terrain.physics_material
#        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
#        # update sensor update periods
#        # we tick all the sensors based on the smallest update period (physics update period)
#        if self.scene.height_scanner is not None:
#            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
#        if self.scene.contact_forces is not None:
#            self.scene.contact_forces.update_period = self.sim.dt
        
        
        self.episode_length_s = 40.0
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        
        self.terminations.base_contact.params["sensor_cfg"].body_names = []
