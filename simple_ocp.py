import pinocchio as pin
import numpy as np
import hppfcl
from numpy import r_, c_, eye
from wrapper_panda import PandaWrapper
from viewer import create_viewer, add_sphere_to_viewer, add_cube_to_viewer

np.set_printoptions(precision=4, linewidth=350, suppress=True, threshold=1e6)
import yaml
from create_ocp import create_ocp

### PARAMETERS
# Number of nodes of the trajectory

# Load variables from the YAML file
with open("config_scenes.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access the 'scene' data
scene = config["scene1"]

TARGET_POSE = pin.SE3(
    pin.utils.rotate(scene["TARGET_POSE"]["rotation"], scene["TARGET_POSE"]["angle"]),
    np.array(scene["TARGET_POSE"]["position"]),
)


#### Creating the robot
robot_wrapper = PandaWrapper()
rmodel, gmodel, vmodel = robot_wrapper()

gmodel.removeGeometryObject("panda1_box_0")
vmodel.removeGeometryObject("panda1_box_0")

radii1 = scene["DIM_OBS"]["radii1"]
radii2 = scene["DIM_ROB"]["radii1"]


Mobs = pin.SE3(
    pin.utils.rotate("y", np.pi) @ pin.utils.rotate("z", np.pi / 2),
    np.array([0, 0.1, 1.2]),
)
rmodel.addFrame(pin.Frame("obstacle", 0, 0, Mobs, pin.OP_FRAME))

idf1 = rmodel.getFrameId("obstacle")
idj1 = rmodel.frames[idf1].parentJoint
elips1 = hppfcl.Ellipsoid(*radii1)
elips1_geom = pin.GeometryObject(
    "el1", idj1, idf1, rmodel.frames[idf1].placement, elips1
)
elips1_geom.meshColor = r_[1, 0, 0, 1]
idg1 = gmodel.addGeometryObject(elips1_geom)

idf2 = rmodel.getFrameId("panda2_hand_tcp")
idj2 = rmodel.frames[idf2].parentJoint
elips2 = hppfcl.Ellipsoid(*radii2)
elips2_geom = pin.GeometryObject(
    "el2", idj2, idf2, rmodel.frames[idf2].placement, elips2
)
elips2_geom.meshColor = r_[1, 1, 0, 1]
idg2 = gmodel.addGeometryObject(elips2_geom)

rdata, gdata = rmodel.createData(), gmodel.createData()

for col_pair in scene["collision_pairs"]:
    gmodel.addCollisionPair(
        pin.CollisionPair(
            gmodel.getGeometryId(col_pair[0]),
            gmodel.getGeometryId(col_pair[1]),
        )
    )


#### Creating the OCP
ocp = create_ocp(rmodel, gmodel, scene)
XS_init = [np.r_[scene["INITIAL_CONFIG"], scene["INITIAL_VELOCITY"]]] * (scene["T"] + 1)
US_init = ocp.problem.quasiStatic(XS_init[:-1])

#### Solving the OCP
ocp.solve(XS_init, US_init, 100)

#### Creating the visualizer
viz = create_viewer(rmodel, gmodel, vmodel)
add_sphere_to_viewer(viz, "goal", 5e-2, TARGET_POSE.translation, color=100000)

#### Displaying the solution
while True:
    for i, xs in enumerate(ocp.xs):
        q = np.array(xs[:7].tolist())
        pin.framesForwardKinematics(rmodel, rdata, q)
        add_cube_to_viewer(
            viz,
            "vcolmpc" + str(i),
            [2e-2, 2e-2, 2e-2],
            rdata.oMf[rmodel.getFrameId("panda2_rightfinger")].translation,
            color=100000000,
        )
        viz.display(np.array(xs[:7].tolist()))
        input()
