from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import TargetChaseVelocityCommandCfg


class TargetChaseVelocityCommand(CommandTerm):
    r"""Drives the robot towards a per-env target on the XY plane.

    Output command is in the robot **base frame** (b): ``[vx_b, vy_b, yaw_rate]``.

    - Linear part points to the target (optionally without strafe).
    - Yaw rate is a P-control on the heading error to the target (Uniform-like semantics).

    If :attr:`cfg.allow_strafe` is False, vy is forced to zero (non-holonomic style).
    """

    cfg: "TargetChaseVelocityCommandCfg"

    def __init__(self, cfg: "TargetChaseVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        # checks like in UniformVelocityCommand
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "TargetChaseVelocityCommand: heading_command=True but `ranges.heading` is None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            # не критично, просто предупреждение как в Uniform
            import omni.log
            omni.log.warn(
                f"TargetChaseVelocityCommand: 'ranges.heading'={self.cfg.ranges.heading} "
                f"but heading_command=False. Consider enabling heading command."
            )

        # assets
        self.robot: Articulation = env.scene[cfg.asset_name]
        try:
            self.target = env.scene[cfg.target_asset_name]
        except KeyError:
            raise KeyError(
                f"[TargetChaseVelocityCommand] target_asset_name '{cfg.target_asset_name}' not found in scene"
            )

        # buffers (совместимо по форме с Uniform)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)   # [vx_b, vy_b, yaw_rate]
        self.heading_target = torch.zeros(self.num_envs, device=self.device)     # world heading to target
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # metrics (как в Uniform)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "TargetChaseVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    # ------------------------
    # Properties
    # ------------------------

    @property
    def command(self) -> torch.Tensor:
        """Desired base velocity command in base frame. Shape: (num_envs, 3)."""
        return self.vel_command_b

    # ------------------------
    # Impl specifics
    # ------------------------

    def _update_metrics(self):
        # как в Uniform — накапливаем средние ошибки
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt if max_command_time > 0.0 else 1.0
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # Мы не сэмплим сами значения команды — только помечаем "стоящие" энвы как в Uniform
        r = torch.empty(len(env_ids), device=self.device)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Compute chase command towards the target in base frame."""
        # --- states (world frame) ---
        base_pos_w = self.robot.data.root_pos_w            # (N,3)
        base_yaw_w = self.robot.data.heading_w             # (N,)
        target_pos_w = self.target.data.root_pos_w         # (N,3)

        # --- geometry to target (world) ---
        delta_w = target_pos_w - base_pos_w
        delta_xy_w = delta_w[:, :2]
        dist_xy = torch.linalg.norm(delta_xy_w, dim=1)

        # world heading to target (для yaw-контроля и метрик)
        self.heading_target = torch.atan2(delta_xy_w[:, 1], delta_xy_w[:, 0])

        # --- linear part in base frame (b) ---
        # поворачиваем в базовую СК по yaw
        cy, sy = torch.cos(base_yaw_w), torch.sin(base_yaw_w)
        dx_b =  cy * delta_xy_w[:, 0] + sy * delta_xy_w[:, 1]
        dy_b = -sy * delta_xy_w[:, 0] + cy * delta_xy_w[:, 1]
        delta_b_xy = torch.stack([dx_b, dy_b], dim=1)

        # профиль скорости
        speed = torch.clamp(self.cfg.k_lin * dist_xy, 0.0, self.cfg.max_speed)
        speed = torch.where(dist_xy < self.cfg.stop_radius, torch.zeros_like(speed), speed)

        if self.cfg.allow_strafe:
            dir_b_xy = torch.nn.functional.normalize(delta_b_xy, dim=1, eps=1e-6)
            vx_b = speed * dir_b_xy[:, 0]
            vy_b = speed * dir_b_xy[:, 1]
        else:
            vx_b = speed
            vy_b = torch.zeros_like(speed)

        # клампы
        vx_b = torch.clamp(vx_b, self.cfg.ranges.lin_vel_x[0], self.cfg.ranges.lin_vel_x[1])
        vy_b = torch.clamp(vy_b, self.cfg.ranges.lin_vel_y[0], self.cfg.ranges.lin_vel_y[1])

        # --- yaw rate (P по ошибке heading) ---
        heading_err = math_utils.wrap_to_pi(self.heading_target - base_yaw_w)
        yaw_rate = torch.clip(
            self.cfg.heading_control_stiffness * heading_err,
            min=self.cfg.ranges.ang_vel_z[0],
            max=self.cfg.ranges.ang_vel_z[1],
        )

        # стоящие энвы — нули
        standing_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        vx_b[standing_ids] = 0.0
        vy_b[standing_ids] = 0.0
        yaw_rate[standing_ids] = 0.0

        # запись
        self.vel_command_b[:, 0] = vx_b
        self.vel_command_b[:, 1] = vy_b
        self.vel_command_b[:, 2] = yaw_rate

    # ------------------------
    # Debug visualization (как в Uniform)
    # ------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # чуть приподнимем стрелки
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # desired/current — берём XY в базе и конвертируем в мир (как в Uniform)
        vel_des_scale, vel_des_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_cur_scale, vel_cur_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_quat, vel_des_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_cur_quat, vel_cur_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert base-frame XY velocity to world-frame arrow (Uniform-style)."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle_b = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle_b)
        arrow_quat_b = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle_b)

        base_quat_w = self.robot.data.root_quat_w
        arrow_quat_w = math_utils.quat_mul(base_quat_w, arrow_quat_b)
        return arrow_scale, arrow_quat_w

