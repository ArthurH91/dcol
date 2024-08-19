import pinocchio as pin
import numpy as np

from wrapper_panda import PandaWrapper
from viewer import create_viewer, add_sphere_to_viewer
from OCP import OCPPandaReachingColWithMultipleCol

### PARAMETERS
# Number of nodes of the trajectory
T = 10
# Time step between each node
dt = 0.01

# OBS CONSTANTS
PLACEMENT_OBS = pin.SE3(pin.utils.rotate("x", 0), np.array([0, -0.2, 1.5]))
DIM_OBS = [0.1, 0.1, 0.1]

# ELLIPS ON THE ROBOT
PLACEMENT_ROB = pin.SE3(pin.utils.rotate("x", 0), np.array([0, 0, 0]))
DIM_ROB =  [0.1, 0.1, 0.1]

# Creating the robot
robot_wrapper = PandaWrapper()
rmodel, cmodel, vmodel = robot_wrapper()

cmodel = robot_wrapper.add_2_ellips(
    cmodel,
    placement_obs=PLACEMENT_OBS,
    dim_obs=DIM_OBS,
    placement_rob=PLACEMENT_ROB,
    dim_rob=DIM_ROB,
)

cmodel.addCollisionPair(
    pin.CollisionPair(cmodel.getGeometryId("ellips_rob"), cmodel.getGeometryId("obstacle"))
)
viz = create_viewer(rmodel, cmodel, vmodel)
INITIAL_CONFIG = pin.neutral(rmodel)

viz.display(INITIAL_CONFIG)


### CREATING THE TARGET
TARGET_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, -0.4, 1.5]))
add_sphere_to_viewer(viz, "goal", 5e-2, TARGET_POSE.translation)


x0 = np.concatenate([INITIAL_CONFIG, pin.utils.zero(rmodel.nv)])

### CREATING THE PROBLEM WITHOUT OBSTACLE
problem = OCPPandaReachingColWithMultipleCol(
    rmodel,
    cmodel,
    TARGET_POSE,
    T,
    dt,
    x0,
    callbacks=True,
    WEIGHT_GRIPPER_POSE=100,
    WEIGHT_GRIPPER_POSE_TERM=500
)
ddp = problem()

ddp.solve()

viz.display(INITIAL_CONFIG)
input()
for xs in ddp.xs:
    viz.display(np.array(xs[:7].tolist()))
    input()
