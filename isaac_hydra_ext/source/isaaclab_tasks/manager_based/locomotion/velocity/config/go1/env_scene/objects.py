from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import RigidObjectCfg, VisualCylinderCfg, CollisionCylinderCfg
import isaaclab.sim as sim_utils

# marker- target
TARGET_MARKER = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Target",
    spawn=sim_utils.CylinderCfg(
        radius=0.08, height=0.02,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.05, 0.9, 0.2), roughness=0.5, metallic=0.0
        ),
    ),
    rigid_props={"kinematic_enabled": True, "disable_gravity": True},
    collision_props={"collision_enabled": False},
    mass_props=None,
)

# obstacle cylynder
OBSTACLE_CYL = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Obstacles/obst_*",
    spawn=sim_utils.CylinderCfg(
        radius=0.12, height=0.60,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.8, 0.8, 0.8), roughness=0.6, metallic=0.0
        ),
    ),
    rigid_props={"kinematic_enabled": True, "disable_gravity": True},
    collision_props={"collision_enabled": True},
    mass_props=None,
)
