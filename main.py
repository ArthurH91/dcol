import argparse
import json

import pinocchio as pin
import numpy as np

from wrapper_panda import PandaWrapper
from viewer import create_viewer, add_sphere_to_viewer
from OCP import OCPPandaReachingColWithMultipleCol
from scenes import Scene

### Parser
parser = argparse.ArgumentParser(description="Parser to select the scenario.")
parser.add_argument(
    "-s", "--save", action="store_true", help="Save the solutions in a .json file."
)
parser.add_argument(
    "-v", "--vel", action="store_true", help="Use the velocity constraint"
)
parser.add_argument("-sc", "--scene", type=int, help="An integer argument")
args = parser.parse_args()

### PARAMETERS
# Number of nodes of the trajectory
T = 10
# Time step between each node
dt = 0.01


### CREATING THE TARGET
TARGET_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0.5, 1.2]))

# Creating the robot
robot_wrapper = PandaWrapper()
rmodel, cmodel, vmodel = robot_wrapper()
scene = Scene()
cmodel = scene.create_scene(cmodel, scene=args.scene)

viz = create_viewer(rmodel, cmodel, vmodel)
# INITIAL_CONFIG = pin.neutral(rmodel)
INITIAL_CONFIG = scene.get_initial_config(scene=args.scene)

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
    max_qp_iters=10000,
    velocity_collision=args.vel,
)
ddp = problem()

ddp.solve()

print("Solved")

if args.save:
    col = "distance"
    if args.vel:
        col = "velocity"

    data = {
        "xs": [array.tolist() for array in ddp.xs.tolist()],
        "us": [array.tolist() for array in ddp.us.tolist()],
        "target": TARGET_POSE.translation.tolist(),
        "obstacle": scene.PLACEMENT_OBS.translation.tolist(),
        "obstacle_dim": scene.DIM_OBS,
        "rob_dim": scene.DIM_ROB,
    }
    with open("results/results" + col + ".json", "w") as json_file:
        json.dump(data, json_file, indent=6)


viz.display(INITIAL_CONFIG)
input()
while True:
    for xs in ddp.xs:
        viz.display(np.array(xs[:7].tolist()))
        input()
