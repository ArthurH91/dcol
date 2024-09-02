import argparse

import pinocchio as pin
import numpy as np

from wrapper_panda import PandaWrapper
from viewer import create_viewer, add_sphere_to_viewer
from OCP import OCPPandaReachingColWithMultipleCol

### Parser
parser = argparse.ArgumentParser(description="Parser to select the scenario.")
parser.add_argument("-s","--save", action="store_true", help="Save the solutions in a .npy file.")
args = parser.parse_args()

### PARAMETERS
# Number of nodes of the trajectory
T = 10
# Time step between each node
dt = 0.01

# OBS CONSTANTS
PLACEMENT_OBS = pin.SE3(pin.utils.rotate("y", np.pi/2) @ pin.utils.rotate("z", np.pi/2), np.array([0, 0.1, 1.2]))
DIM_OBS = [0.1, 0.2, 0.1]

# ELLIPS ON THE ROBOT
PLACEMENT_ROB = pin.SE3(np.eye(3), np.array([0, 0, 0.]))
DIM_ROB =  [0.1, 0.1, 0.17]

### CREATING THE TARGET
TARGET_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0.5, 1.2]))

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
# INITIAL_CONFIG = pin.neutral(rmodel)
INITIAL_CONFIG = np.array([-0.06709294 , 1.35980773 ,-0.81605989,  0.74243348,  0.42419277 , 0.45547585, -0.00456262])
viz.display(INITIAL_CONFIG)
add_sphere_to_viewer(viz, "goal", 5e-2, TARGET_POSE.translation, color=100000)


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
    WEIGHT_GRIPPER_POSE=200,
    WEIGHT_GRIPPER_POSE_TERM=1000,
    max_qp_iters=10000
)
ddp = problem()

ddp.solve()

print("Solved")

if args.save:
    np.save("results",np.concatenate(np.array(ddp.xs.tolist() + ddp.us.tolist())))


viz.display(INITIAL_CONFIG)
input()
while True:
    for xs in ddp.xs:
        viz.display(np.array(xs[:7].tolist()))
        input() 
