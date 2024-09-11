import pinocchio as pin
import numpy as np
import hppfcl
from wrapper_panda import PandaWrapper

import numdifftools as nd

def add_ellipsoid(
    cmodel: pin.GeometryModel,
    name: str,
    parentJoint=0,
    parentFrame=0,
    placement=pin.SE3.Random(),
    dim=[0.2, 0.5, 0.1],
) -> pin.GeometryModel:
    """
    Add an ellipsoid geometry object to the given `cmodel`.

    Args:
        cmodel (pin.GeometryModel): The geometry model to add the ellipsoid to.
        name (str): The name of the ellipsoid.
        parentJoint (int, optional): The index of the parent joint. Defaults to 0.
        parentFrame (int, optional): The index of the parent frame. Defaults to 0.
        placement (pin.SE3, optional): The placement of the ellipsoid. Defaults to pin.SE3.Random().
        dim (List[float], optional): The dimensions of the ellipsoid [x, y, z]. Defaults to [0.2, 0.5, 0.1].

    Returns:
        pin.GeometryModel: The updated geometry model with the added ellipsoid.
    """

    elips = hppfcl.Ellipsoid(dim[0], dim[1], dim[2])
    elips_geom = pin.GeometryObject(
        name,
        parent_joint=parentJoint,
        parent_frame=parentFrame,
        collision_geometry=elips,
        placement=placement,
    )
    elips_geom.meshColor = np.concatenate(
        (np.random.uniform(0, 1, 3), np.ones(1) / 0.8)
    )

    cmodel.addGeometryObject(elips_geom)
    return cmodel


def cp(q, rmodel, rdata, cmodel, cdata):
    """Computing the closest points in each shape "obstacle" and "ellips_rob"
    """

    # Updating the position of the joints & the geometry objects.
    pin.forwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)

    # Getting the geometry of the shape 1
    shape1_geom = shape1.geometry
    # Getting its pose in the world reference
    shape1_placement = cdata.oMg[shape1_id]
    # Doing the same for the second shape.
    shape2_geom = shape2.geometry
    shape2_placement = cdata.oMg[shape2_id]

    # Compute distances and nearest points using HPP-FCL
    req = hppfcl.DistanceRequest()
    req.gjk_max_iterations = 20000
    req.abs_err = 0
    req.gjk_tolerance = 1e-9
    res = hppfcl.DistanceResult()
    _ = hppfcl.distance(
        shape1_geom, shape1_placement, shape2_geom, shape2_placement, req, res
    )
    cp1 = res.getNearestPoint1()
    cp2 = res.getNearestPoint2()
    return np.concatenate((cp1, cp2))


def get_dx_dtheta(sol_x1, sol_x2,c1, c2, A1, A2, sol_lam1, sol_lam2):

    Lyy = np.r_[
        np.c_[np.eye(3) + sol_lam1 * A1, -np.eye(3), A1 @ (sol_x1 - c1), np.zeros(3)],
        np.c_[-np.eye(3), np.eye(3) + sol_lam2 * A2, np.zeros(3), A2 @ (sol_x2 - c2)],
        [np.r_[A1 @ (sol_x1 - c1), np.zeros(3), np.zeros(2)]],
        [np.r_[np.zeros(3), A2 @ (sol_x2 - c2), np.zeros(2)]],
    ]
    Lyc = np.r_[
        np.c_[-sol_lam1 * A1, np.zeros([3, 3])],
        np.c_[np.zeros([3, 3]), -sol_lam2 * A2],
        [np.r_[-A1 @ (sol_x1 - c1), np.zeros(3)]],
        [np.r_[np.zeros(3), -A2 @ (sol_x2 - c2)]],
    ]
    Lyr = np.r_[
        np.c_[
            sol_lam1 * (A1 @ pin.skew(sol_x1 - c1) - pin.skew(A1 @ (sol_x1 - c1))),
            np.zeros([3, 3]),
        ],
        np.c_[
            np.zeros([3, 3]),
            sol_lam2 * (A2 @ pin.skew(sol_x2 - c2) - pin.skew(A2 @ (sol_x2 - c2))),
        ],
        [np.r_[(sol_x1 - c1) @ A1 @ pin.skew(sol_x1 - c1), np.zeros(3)]],
        [np.r_[np.zeros(3), (sol_x2 - c2) @ A2 @ pin.skew(sol_x2 - c2)]],
    ]
    # Tolerance here must be very high, not sure why, but those tests are not super important

    # ##############################

    yc = -np.linalg.inv(Lyy) @ Lyc
    yr = -np.linalg.inv(Lyy) @ Lyr

    # ####################
    # Check with DDL

    dx1 = np.c_[yc[:3], yr[:3]]
    dx2 = np.c_[yc[3:6], yr[3:6]]

    return dx1, dx2


def compute_dx_dtheta(q, rmodel, rdata, cmodel, cdata):
    
    # Updating the position of the joints & the geometry objects.
    pin.forwardKinematics(rmodel, rdata, q)
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)

    # Poses and geometries of the shapes
    shape1_placement = cdata.oMg[shape1_id]
    shape2_placement = cdata.oMg[shape2_id]

    req = hppfcl.DistanceRequest()
    res = hppfcl.DistanceResult()
    _ = hppfcl.distance(
        shape1.geometry,
        shape1_placement,
        shape2.geometry,
        shape2_placement,
        req,
        res,
    )
    x1 = res.getNearestPoint1()
    x2 = res.getNearestPoint2()
    
    c1 = shape1_placement.translation
    c2 = shape2_placement.translation


    D1 = np.diagflat(shape1.geometry.radii)
    D2 = np.diagflat(shape2.geometry.radii)

    R1 = shape1_placement.rotation
    R2 = shape2_placement.rotation
    A1, A2 = R1 @ D1 @ R1.T, R2 @ D2 @ R2.T  # From pinocchio A = RDR.T

    sol_lam1, sol_lam2 = -(x1 - c1).T @ (x1 - x2), (x2 - c2).T @ (x1 - x2)

    dx1, dx2 = get_dx_dtheta(
        x1, x2, c1, c2, A1, A2, sol_lam1, sol_lam2
    )
    
    dtheta1_dq = pin.computeFrameJacobian(
        rmodel, rdata,q, shape1.parentFrame, pin.LOCAL_WORLD_ALIGNED
    )
    dc1_dq = dtheta1_dq[:3][:]
    dr1_dq = dtheta1_dq[3:][:]
    
    dtheta2_dq = pin.computeFrameJacobian(
        rmodel, rdata, q, shape2.parentFrame, pin.LOCAL_WORLD_ALIGNED
    )
    
    dc2_dq = dtheta2_dq[:3][:]
    dr2_dq = dtheta2_dq[3:][:]
    
    dtheta_dq = np.r_[dc1_dq, dc2_dq, dr1_dq, dr2_dq]
    
    dx1_dq = dx1 @ dtheta_dq
    dx2_dq = dx2 @ dtheta_dq
    
    return dx1_dq, dx2_dq

def numdiff_matrix(f, inX, h=1e-6):
    f0 = np.array(f(inX)).copy()
    x = inX.copy()
    df_dx = np.zeros((f0.size, len(x)))
    for ix in range(len(x)):
        x[ix] += h
        df_dx[:, ix] = (f(x) - f0) / h
        x[ix] = inX[ix]
    return df_dx

def c1(q):
    pin.forwardKinematics(rmodel, rdata, q)
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    c = cdata.oMg[shape1_id].translation
    return c

def c2(q):
    pin.forwardKinematics(rmodel, rdata, q)
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    c = cdata.oMg[shape2_id].translation
    return c

def r1(q):
    pin.forwardKinematics(rmodel, rdata, q)
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    r = cdata.oMg[shape1_id].rotation
    return r

def r2(q):
    pin.forwardKinematics(rmodel, rdata, q)
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    r = cdata.oMg[shape2_id].rotation
    return r





###### MAIN ######
robot_wrapper = PandaWrapper()
rmodel, cmodel, vmodel = robot_wrapper()

PLACEMENT_OBS = pin.SE3(pin.SE3.Random().rotation, np.random.uniform(-100, 100, 3))
DIM_OBS = [0.25, 0.1, 0.3]

PLACEMENT_ROB = pin.SE3(pin.SE3.Random().rotation, np.random.uniform(-100, 100, 3))
DIM_ROB = [0.1, 0.9, 0.5]


#### Creating the shapes
parent_frame_obs = 0
parent_joint_obs = 0
parent_frame_rob = cmodel.geometryObjects[
    cmodel.getGeometryId("panda2_link7_sc_5")
].parentFrame
parent_joint_rob = cmodel.geometryObjects[
    cmodel.getGeometryId("panda2_link7_sc_5")
].parentJoint

add_ellipsoid(cmodel, "obstacle", placement=PLACEMENT_OBS, dim=DIM_OBS)
add_ellipsoid(
    cmodel,
    "ellips_rob",
    parentJoint=parent_joint_rob,
    parentFrame=parent_frame_rob,
    placement=PLACEMENT_ROB,
    dim=DIM_ROB,
)


#### Getting the shapes
shape1_id= cmodel.getGeometryId("obstacle")
shape2_id = cmodel.getGeometryId("ellips_rob")

shape1 = cmodel.geometryObjects[shape1_id]
shape2 = cmodel.geometryObjects[shape2_id]


#### Creating the data
rdata = rmodel.createData()
cdata = cmodel.createData()


#### Random configurations
q = pin.randomConfiguration(rmodel)
v = pin.randomConfiguration(rmodel)


#### Forward kinematics
pin.forwardKinematics(rmodel, rdata, q, v)
pin.framesForwardKinematics(rmodel, rdata, q)
pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
pin.computeForwardKinematicsDerivatives(rmodel, rdata, q, v, np.zeros(rmodel.nv))



#### Computing the closest points and their derivatives
dx1_dq, dx2_dq = compute_dx_dtheta(q, rmodel, rdata, cmodel, cdata)
dcp_dq = nd.Gradient(lambda variable: cp(variable, rmodel, rdata, cmodel, cdata))(q)
dcp_dq_ = numdiff_matrix(lambda variable: cp(variable, rmodel, rdata, cmodel, cdata), q)

# print(f"dx1_dq: {dx1_dq}")
# print(f"dcp1_dq: {dcp_dq[:3]}")
# print(f"dx2_dq: {dx2_dq}")
# print(f"dcp2_dq: {dcp_dq[3:]}")
print(f"dcp1_q - dx1_dq: {dcp_dq[:3] - dx1_dq}")
print(f"dcp2_q - dx2_dq: {dcp_dq[3:] - dx2_dq}")



#### Computing the derivatives of theta with respect to q
dtheta1_dq = pin.computeFrameJacobian(rmodel, rdata, q, parent_frame_obs, pin.LOCAL_WORLD_ALIGNED)
dtheta2_dq = pin.computeFrameJacobian(rmodel, rdata, q, parent_frame_rob, pin.LOCAL_WORLD_ALIGNED)

dc1_dq = dtheta1_dq[:3][:]
dc2_dq = dtheta2_dq[:3][:]
dr1_dq = dtheta1_dq[3:][:]
dr2_dq = dtheta2_dq[3:][:]

dc1_dq_nd = nd.Gradient(c1)(q)
dc2_dq_nd = nd.Gradient(c2)(q)
dr1_dq_nd = nd.Gradient(r1)(q)
dr2_dq_nd = nd.Gradient(r2)(q)

    
print(dc1_dq - dc1_dq_nd)

print(dc2_dq - dc2_dq_nd)
# print(dc2_dq)
# print(dc2_dq_nd)