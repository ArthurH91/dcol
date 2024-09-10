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
    "-d", "--disablecol", action="store_true", help="Do not use collision avoidance."
)
parser.add_argument(
    "-v", "--vel", action="store_true", help="Use the velocity constraint"
)
parser.add_argument("-sc", "--scene", type=int, help="An integer argument")
args = parser.parse_args()



args.vel = True
args.scene = 4
print(args.disablecol)


### PARAMETERS
# Number of nodes of the trajectory
T = 20
# Time step between each node
dt = 0.05

# Creating the robot
robot_wrapper = PandaWrapper()
rmodel, cmodel, vmodel = robot_wrapper()
cmodel.removeGeometryObject("panda1_box_0")
vmodel.removeGeometryObject("panda1_box_0")

scene = Scene()
cmodel = scene.create_scene(cmodel, scene=args.scene)

rdata = rmodel.createData()
viz = create_viewer(rmodel, cmodel, vmodel)
INITIAL_CONFIG = scene.get_initial_config(scene=args.scene)
TARGET_POSE = scene.get_target_pose(scene=args.scene)
hyperparams = scene.get_hyperparams(scene=args.scene)
viz.display(INITIAL_CONFIG)
add_sphere_to_viewer(viz, "goal", 5e-2, TARGET_POSE.translation, color=100000)


x0 = np.concatenate([INITIAL_CONFIG, pin.utils.zero(rmodel.nv)])

# # ### CREATING THE PROBLEM WITHOUT OBSTACLE
# problem = OCPPandaReachingColWithMultipleCol(
#     rmodel,
#     cmodel,
#     TARGET_POSE,
#     T,
#     dt,
#     x0,
#     callbacks=True,
#     WEIGHT_xREG=5e-2,
#     WEIGHT_GRIPPER_POSE=10,
#     WEIGHT_GRIPPER_POSE_TERM=100,
#     max_qp_iters=10000,
#     disable_collision=args.disablecol,
#     velocity_collision=False,
#     SAFETY_THRESHOLD=0,
#     ksi = scene.ksi,
#     di = scene.di,
#     ds = scene.ds
# )
# ddp = problem()


# ddp.solve()

# for i, xs in enumerate(ddp.xs):
#     q = np.array(xs[:7].tolist())
#     pin.framesForwardKinematics(rmodel, rdata, q)
#     add_sphere_to_viewer(viz, "colmpc" + str(i), 1e-2, rdata.oMf[rmodel.getFrameId("panda2_rightfinger")].translation, color=1000000)
# xs = ddp.xs
# us = ddp.us

# ddp.solve()

problem = OCPPandaReachingColWithMultipleCol(
    rmodel,
    cmodel,
    TARGET_POSE,
    T,
    dt,
    x0,
    callbacks=True,
    WEIGHT_xREG=5e-2,
    WEIGHT_GRIPPER_POSE=10,
    WEIGHT_GRIPPER_POSE_TERM=1000,
    max_qp_iters=10000,
    disable_collision=args.disablecol,
    velocity_collision=args.vel,
    SAFETY_THRESHOLD=0,
    ksi = scene.ksi,
    di = scene.di,
    ds = scene.ds
)
ddp = problem()


# ddp.solve(xs, us, 100)

ddp.solve()

print("Solved")

onstacles_pose_translation = [placement.translation.tolist() for placement in scene.PLACEMENT_OBS]
onstacles_pose_rotation = [placement.rotation.tolist() for placement in scene.PLACEMENT_OBS]

dims_obs = [obs for obs in scene.DIM_OBS]
dims_rob = [rob for rob in scene.DIM_ROB]

collision_pairs = [[col.first, col.second] for col in cmodel.collisionPairs]

if args.save:
    col = "distance"
    if args.vel:
        col = "velocity"

    data = {
        "xs": [array.tolist() for array in ddp.xs.tolist()],
        "us": [array.tolist() for array in ddp.us.tolist()],
        "target_pose_translation": TARGET_POSE.translation.tolist(),
        "target_pose_rotation": TARGET_POSE.rotation.tolist(),
        "obstacle_pose_translation": onstacles_pose_translation,
        "obstacle_pose_rotation": onstacles_pose_rotation,
        "obstacle_dim": dims_obs,
        "rob_dim": dims_rob,
        "T": T,
        "collision_pairs": collision_pairs,
    }
    if args.disablecol:
        name = "resultsnocol"
    else:
        name = "results" + col
    with open("results/scene"+ str(args.scene) + "/" + name + ".json", "w") as json_file:
        json.dump(data, json_file, indent=6)
    print("saved in", "results/scene"+ str(args.scene) + "/" + name + ".json",)

viz.display(INITIAL_CONFIG)
input()
while True:
    for i,xs in enumerate(ddp.xs):
        q = np.array(xs[:7].tolist())
        pin.framesForwardKinematics(rmodel, rdata, q)
        add_sphere_to_viewer(viz, "vcolmpc" + str(i), 1e-2, rdata.oMf[rmodel.getFrameId("panda2_rightfinger")].translation, color=100000000)
        viz.display(np.array(xs[:7].tolist()))
        input()