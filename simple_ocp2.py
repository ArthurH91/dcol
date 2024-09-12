import pinocchio as pin
import numpy as np
import hppfcl
import crocoddyl
import mim_solvers
from numpy import r_, c_, eye
from wrapper_panda import PandaWrapper
from viewer import create_viewer, add_sphere_to_viewer, add_cube_to_viewer
np.set_printoptions(precision=4, linewidth=350, suppress=True,threshold=1e6)

from VR import ResidualModelVelocityAvoidance
from colmpc import ResidualDistanceCollision
### PARAMETERS
# Number of nodes of the trajectory
T = 30
# Time step between each node
dt = 0.1



INITIAL_CONFIG = np.array([
                    6.87676046e-02,
                    1.87133260e00,
                    -9.23646871e-01,
                    6.62962572e-01,
                    5.02801754e-01,
                    1.696128891e-00,
                    4.77514312e-01,
                ])
INITIAL_VELOCITY = np.zeros(7)
x0 = np.r_[INITIAL_CONFIG, INITIAL_VELOCITY]
TARGET_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0.2, 0.9]))

ksi = 1e-1
di = 1e-1 #1e-4
ds = 1e-7

WEIGHT_uREG=1e-4
WEIGHT_xREG=1e-1
WEIGHT_GRIPPER_POSE=50
WEIGHT_GRIPPER_POSE_TERM=100
WEIGHT_LIMIT=1e-1
SAFETY_THRESHOLD=0

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

#### Creating the visualizer
viz = create_viewer(rmodel, gmodel, vmodel)

viz.display(INITIAL_CONFIG)
add_sphere_to_viewer(viz, "goal", 5e-2, TARGET_POSE.translation, color=100000)


#### Creating the OCP 

# Stat and actuation model
state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFull(state)

# Running & terminal cost models
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

### Creation of cost terms

# State Regularization cost
xResidual = crocoddyl.ResidualModelState(state, x0)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)

# Control Regularization cost
uResidual = crocoddyl.ResidualModelControl(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

# End effector frame cost
framePlacementResidual = crocoddyl.ResidualModelFrameTranslation(
    state,
    rmodel.getFrameId("panda2_rightfinger"),
    TARGET_POSE.translation,
)

goalTrackingCost = crocoddyl.CostModelResidual(
    state, framePlacementResidual
)

# Obstacle cost with hard constraint
runningConstraintModelManager = crocoddyl.ConstraintModelManager(
    state, actuation.nu
)
terminalConstraintModelManager = crocoddyl.ConstraintModelManager(
    state, actuation.nu
)
# Creating the residual
    
obstacleDistanceResidual = ResidualModelVelocityAvoidance(
state, 1 ,gmodel, 0, di, ds, ksi
)
for col_idx, col_pair in enumerate(gmodel.collisionPairs):
    obstaclVelocityResidual = ResidualModelVelocityAvoidance(
    state ,gmodel, col_pair.first, col_pair.second,ksi, di, ds, 
    )

    # Creating the inequality constraint
    constraint = crocoddyl.ConstraintModelResidual(
        state,
        obstaclVelocityResidual,
        np.array([0]),
        np.array([np.inf]),
    )

    # Adding the constraint to the constraint manager
    runningConstraintModelManager.addConstraint(
        "col_" + str(col_idx), constraint
    )
    terminalConstraintModelManager.addConstraint(
        "col_term_" + str(col_idx), constraint
    )
    obstaclVelocityResidual = ResidualModelVelocityAvoidance(
    state ,gmodel, col_pair.first, col_pair.second,ksi, di, ds, 
    )

# Adding costs to the models
runningCostModel.addCost("stateReg", xRegCost, WEIGHT_xREG)
runningCostModel.addCost("ctrlRegGrav", uRegCost, WEIGHT_uREG)
runningCostModel.addCost(
    "gripperPoseRM", goalTrackingCost, WEIGHT_GRIPPER_POSE
)
terminalCostModel.addCost("stateReg", xRegCost, WEIGHT_xREG)
terminalCostModel.addCost(
    "gripperPose", goalTrackingCost, WEIGHT_GRIPPER_POSE_TERM
)

# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
    state,
    actuation,
    runningCostModel,
    runningConstraintModelManager,
)
terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
    state,
    actuation,
    terminalCostModel,
    terminalConstraintModelManager,
)

runningModel = crocoddyl.IntegratedActionModelEuler(
    running_DAM, dt
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    terminal_DAM, 0.0
)


problem = crocoddyl.ShootingProblem(
    x0, [runningModel] * T, terminalModel
)
# Create solver + callbacks
# Define mim solver with inequalities constraints
ddp = mim_solvers.SolverCSQP(problem)

# Merit function
ddp.use_filter_line_search = False

# Parameters of the solver
ddp.termination_tolerance = 1e-3
ddp.max_qp_iters = 10000
ddp.eps_abs = 1e-6
ddp.eps_rel = 0

ddp.with_callbacks = True
XS_init = [x0] * (T+1)
US_init = ddp.problem.quasiStatic(XS_init[:-1])


ddp.solve(XS_init, US_init, 100)


while True:
    for i,xs in enumerate(ddp.xs):
        q = np.array(xs[:7].tolist())
        pin.framesForwardKinematics(rmodel, rdata, q)
        add_cube_to_viewer(viz, "vcolmpc" + str(i), [2e-2,2e-2, 2e-2], rdata.oMf[rmodel.getFrameId("panda2_rightfinger")].translation, color=100000000)
        viz.display(np.array(xs[:7].tolist()))
        input()