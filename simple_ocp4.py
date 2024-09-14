import pinocchio as pin
import numpy as np
import hppfcl
from numpy import r_
from wrapper_panda import PandaWrapper
from viewer import create_viewer, add_sphere_to_viewer, add_cube_to_viewer
np.set_printoptions(precision=4, linewidth=350, suppress=True,threshold=1e6)

import yaml
from create_ocp import create_ocp_velocity, create_ocp_distance, create_ocp_nocol

# Load variables from the YAML file
with open("config_scenes.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access the 'scene' data
scene = config["scene4"]

#### Creating the robot
robot_wrapper = PandaWrapper()
rmodel, gmodel, vmodel = robot_wrapper()

radii1 = scene["DIM_OBS"][0]
radii2 = scene["DIM_ROB"][0]


gmodel.removeGeometryObject("panda1_box_0")
vmodel.removeGeometryObject("panda1_box_0")


Mobs = pin.SE3(pin.utils.rotate("y", np.pi ) @ pin.utils.rotate("z", np.pi / 2),np.array([0, 0.1, 1.2]))
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
rdata,gdata = rmodel.createData(),gmodel.createData()

gmodel.addCollisionPair(
                pin.CollisionPair(
                    gmodel.getGeometryId("el1"),
                    gmodel.getGeometryId("el2"),
                ))

#### Creating the OCP for the velocity
ocp_vel = create_ocp_velocity(rmodel, gmodel, scene)
XS_init = [np.r_[scene["INITIAL_CONFIG"], scene["INITIAL_VELOCITY"]]] * (scene["T"] + 1)
US_init = ocp_vel.problem.quasiStatic(XS_init[:-1])

#### Solving the OCP
ocp_vel.solve(XS_init, US_init, 100)

# #### Creating the OCP for the distance
ocp_dist = create_ocp_distance(rmodel, gmodel, scene)
ocp_dist.solve(XS_init,US_init, 100)

### Creating the OCP for the no collision
ocp_nocol = create_ocp_nocol(rmodel, gmodel, scene)
ocp_nocol.solve(XS_init, US_init, 100)
print(ocp_nocol.xs.tolist()[-1])
#### Creating the visualizer
viz = create_viewer(rmodel, gmodel, vmodel)
add_sphere_to_viewer(viz, "goal", 5e-2,  np.array(scene["TARGET_POSE"]["translation"]), color=100000)

#### Displaying the trajectories

for i, xs in enumerate(ocp_vel.xs):
        q = np.array(xs[:7].tolist())
        pin.framesForwardKinematics(rmodel, rdata, q)
        add_cube_to_viewer(
            viz,
            "vcolmpc" + str(i),
            [2e-2, 2e-2, 2e-2],
            rdata.oMf[rmodel.getFrameId("panda2_rightfinger")].translation,
            color=100000000,
        )

for i, xs in enumerate(ocp_dist.xs):
        q = np.array(xs[:7].tolist())
        pin.framesForwardKinematics(rmodel, rdata, q)
        add_sphere_to_viewer(
            viz,
            "colmpc" + str(i),
            2e-2,
            rdata.oMf[rmodel.getFrameId("panda2_rightfinger")].translation,
            color=100000,
        )

from save_results import save_results
save_results(ocp_dist, ocp_vel, ocp_nocol, scene_nb=4)
        

#### Displaying the solution
while True:
    for i, xs in enumerate(ocp_vel.xs):
        q = np.array(xs[:7].tolist())
        pin.framesForwardKinematics(rmodel, rdata, q)
        viz.display(np.array(xs[:7].tolist()))
        input()
