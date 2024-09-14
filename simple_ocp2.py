import pinocchio as pin
import numpy as np
import hppfcl
from numpy import r_, c_, eye
from wrapper_panda import PandaWrapper
from viewer import create_viewer, add_sphere_to_viewer, add_cube_to_viewer
np.set_printoptions(precision=4, linewidth=350, suppress=True,threshold=1e6)

import yaml
from create_ocp import create_ocp_velocity, create_ocp_distance

# Load variables from the YAML file
with open("config_scenes.yaml", "r") as file:
    config = yaml.safe_load(file)

# Access the 'scene' data
scene = config["scene2"]


#### Creating the robot
robot_wrapper = PandaWrapper()
rmodel, gmodel, vmodel = robot_wrapper()

gmodel.removeGeometryObject("panda1_box_0")
vmodel.removeGeometryObject("panda1_box_0")


PLACEMENT_OBS = [
pin.SE3(pin.utils.rotate("y", np.pi / 2), np.array([0.0, 0.0, 0.9])),
pin.SE3(pin.utils.rotate("y", np.pi / 2), np.array([0.0, 0.4, 0.9])),
pin.SE3(
    pin.utils.rotate("y", np.pi / 2) @ pin.utils.rotate("x", np.pi / 2),
    np.array([-0.2, 0.2, 0.9]),
),
pin.SE3(
    pin.utils.rotate("y", np.pi / 2) @ pin.utils.rotate("x", np.pi / 2),
    np.array([0.2, 0.2, 0.9]),
),
]
DIM_OBS = [[0.09, 0.06, 0.2], [0.09, 0.06, 0.2], [0.09, 0.06, 0.2], [0.09, 0.06, 0.2]]
DIM_ROB = [[0.1, 0.08, 0.15], [0.04, 0.06, 0.04]]

rmodel.addFrame(pin.Frame('obstacle_1',0,0,PLACEMENT_OBS[0],pin.OP_FRAME))
rmodel.addFrame(pin.Frame('obstacle_2',0,0,PLACEMENT_OBS[1],pin.OP_FRAME))
rmodel.addFrame(pin.Frame('obstacle_3',0,0,PLACEMENT_OBS[2],pin.OP_FRAME))
rmodel.addFrame(pin.Frame('obstacle_4',0,0,PLACEMENT_OBS[3],pin.OP_FRAME))

idf_obs_1 = rmodel.getFrameId('obstacle_1')
idj_obs_1 = rmodel.frames[idf_obs_1].parentJoint
elips_obs_1 = hppfcl.Ellipsoid(*DIM_OBS[0])
elips_obs_1_geom = pin.GeometryObject('el_obs_1', idj_obs_1,idf_obs_1,rmodel.frames[idf_obs_1].placement,elips_obs_1)
elips_obs_1_geom.meshColor = r_[1,0,0,1]
idg_obs_1 = gmodel.addGeometryObject(elips_obs_1_geom)

idf_obs_2 = rmodel.getFrameId('obstacle_2')
idj_obs_2 = rmodel.frames[idf_obs_2].parentJoint
elips_obs_2 = hppfcl.Ellipsoid(*DIM_OBS[0])
elips_obs_2_geom = pin.GeometryObject('el_obs_2', idj_obs_2,idf_obs_2,rmodel.frames[idf_obs_2].placement,elips_obs_2)
elips_obs_2_geom.meshColor = r_[1,0,0,1]
idg_obs_2 = gmodel.addGeometryObject(elips_obs_2_geom)

idf_obs_3 = rmodel.getFrameId('obstacle_3')
idj_obs_3 = rmodel.frames[idf_obs_3].parentJoint
elips_obs_3 = hppfcl.Ellipsoid(*DIM_OBS[0])
elips_obs_3_geom = pin.GeometryObject('el_obs_3', idj_obs_3,idf_obs_3,rmodel.frames[idf_obs_3].placement,elips_obs_3)
elips_obs_3_geom.meshColor = r_[1,0,0,1]
idg_obs_3 = gmodel.addGeometryObject(elips_obs_3_geom)

idf_obs_4 = rmodel.getFrameId('obstacle_4')
idj_obs_4 = rmodel.frames[idf_obs_4].parentJoint
elips_obs_4 = hppfcl.Ellipsoid(*DIM_OBS[0])
elips_obs_4_geom = pin.GeometryObject('el_obs_4', idj_obs_4,idf_obs_4,rmodel.frames[idf_obs_4].placement,elips_obs_4)
elips_obs_4_geom.meshColor = r_[1,0,0,1]
idg_obs_4 = gmodel.addGeometryObject(elips_obs_4_geom)

idf_rob_1 = rmodel.getFrameId('panda2_link7_sc')
idj_rob_1 = rmodel.frames[idf_rob_1].parentJoint
elips_rob_1 = hppfcl.Ellipsoid(*DIM_ROB[0])
elips_rob_1_geom = pin.GeometryObject('el_rob_1', idj_rob_1,idf_rob_1,rmodel.frames[idf_rob_1].placement,elips_rob_1)
elips_rob_1_geom.meshColor = r_[1,1,0,1]
idg_rob_1 = gmodel.addGeometryObject(elips_rob_1_geom)


rdata,gdata = rmodel.createData(),gmodel.createData()

gmodel.addCollisionPair(
                pin.CollisionPair(
                    gmodel.getGeometryId("el_obs_1"),
                    gmodel.getGeometryId("el_rob_1"),
                ))

gmodel.addCollisionPair(
                pin.CollisionPair(
                    gmodel.getGeometryId("el_obs_2"),
                    gmodel.getGeometryId("el_rob_1"),
                ))

gmodel.addCollisionPair(
                pin.CollisionPair(
                    gmodel.getGeometryId("el_obs_3"),
                    gmodel.getGeometryId("el_rob_1"),
                ))
gmodel.addCollisionPair(
                pin.CollisionPair(
                    gmodel.getGeometryId("el_obs_4"),
                    gmodel.getGeometryId("el_rob_1"),
                ))


#### Creating the OCP for the velocity
ocp_vel = create_ocp_velocity(rmodel, gmodel, scene)
XS_init = [np.r_[scene["INITIAL_CONFIG"], scene["INITIAL_VELOCITY"]]] * (scene["T"] + 1)
US_init = ocp_vel.problem.quasiStatic(XS_init[:-1])

#### Solving the OCP
ocp_vel.solve(XS_init, US_init, 100)

#### Creating the OCP for the distance
ocp_dist = create_ocp_distance(rmodel, gmodel, scene)
ocp_dist.solve(XS_init,US_init, 100)

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

#### Displaying the solution
while True:
    for i, xs in enumerate(ocp_dist.xs):
        q = np.array(xs[:7].tolist())
        pin.framesForwardKinematics(rmodel, rdata, q)
        viz.display(np.array(xs[:7].tolist()))
        input()
