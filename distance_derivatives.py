### This file defines all the functions used for the distance computation and its derivatives.


import numpy as np
import pinocchio as pin
import hppfcl

from derivatives_computation import DerivativeComputation


def dist(rmodel, cmodel, q):
    """Computing the distance between the two shapes of the robot.

    Args:
        rmodel (_type_): _description_
        cmodel (_type_): _description_
        q (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Creating the data models
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Updating the position of the joints & the geometry objects.
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)

    # Poses and geometries of the shapes
    shape1_id = cmodel.getGeometryId("obstacle")
    shape1 = cmodel.geometryObjects[shape1_id]

    shape2_id = cmodel.getGeometryId("ellips_rob")
    shape2 = cmodel.geometryObjects[shape2_id]

    # Getting the geometry of the shape 1
    shape1_geom = shape1.geometry
    # Getting its pose in the world reference
    shape1_placement = cdata.oMg[shape1_id]
    # Doing the same for the second shape.
    shape2_geom = shape2.geometry
    shape2_placement = cdata.oMg[shape2_id]

    req = hppfcl.DistanceRequest()
    res = hppfcl.DistanceResult()
    dist = hppfcl.distance(
        shape1_geom,
        shape1_placement,
        shape2_geom,
        shape2_placement,
        req,
        res,
    )
    return dist


def cp(rmodel, cmodel, q):
    """Computing the closest points in each shape "obstacle" and "ellips_rob".

    Args:
        rmodel (_type_): _description_
        cmodel (_type_): _description_
        q (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Creating the data models
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Updating the position of the joints & the geometry objects.
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)

    # Poses and geometries of the shapes
    shape1_id = cmodel.getGeometryId("obstacle")
    shape1 = cmodel.geometryObjects[shape1_id]

    shape2_id = cmodel.getGeometryId("ellips_rob")
    shape2 = cmodel.geometryObjects[shape2_id]

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

def R(rmodel, cmodel, q, shape_name):
    # Creating the data models
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Updating the position of the joints & the geometry objects.
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    # Poses and geometries of the shapes
    shape_id = cmodel.getGeometryId(shape_name)
    # Getting its pose in the world reference
    shape_placement = cdata.oMg[shape_id]
    return shape_placement.rotation


def dR_dt(rmodel, cmodel, x, shape_name):
           
    q = x[: rmodel.nq]
    v = x[rmodel.nq :]
    
    # Creating the data models
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Updating the position of the joints & the geometry objects.
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    pin.forwardKinematics(rmodel, rdata, q, v)
    pin.updateFramePlacements(rmodel, rdata)

    # Poses and geometries of the shapes
    shape_id = cmodel.getGeometryId(shape_name)
    shape = cmodel.geometryObjects[shape_id]

    # Getting its pose in the world reference
    R = cdata.oMg[shape_id].rotation
    v = pin.getFrameVelocity(rmodel, rdata, shape.parentFrame, pin.LOCAL_WORLD_ALIGNED)

    wx = pin.skew(v.angular)
    
    return wx @ R


def A(rmodel, cmodel, q, shape_name):
    """Returns the matrix A that is the matrix defining the geometry and the rotation of the ellipsoid.
    A = R.T @ D @ R. Where R is the rotation matrix of the ellipsoid and D is the radii matrix.

    Args:
        rmodel (_type_): _description_
        cmodel (_type_): _description_
        q (_type_): _description_
        shape_name (_type_): _description_

    Returns:
        A1
    """
    # Poses and geometries of the shapes
    shape_id = cmodel.getGeometryId(shape_name)
    shape = cmodel.geometryObjects[shape_id]
    Rot = R(rmodel, cmodel, q, shape_name=shape_name)
    shape_geom = shape.geometry  # Get the frame geom
    shape_radii = shape_geom.radii # Get the radii of shape 1
    D = np.array(
            [
                [1 / shape_radii[0] ** 2, 0, 0],
                [0, 1 / shape_radii[1] ** 2, 0],
                [0, 0, 1 / shape_radii[2] ** 2],
            ]
        )
    A = Rot.T @ D @ Rot

    return A

def dA_dt(rmodel, cmodel, x, shape_name):
    q = x[: rmodel.nq]
    
    # Poses and geometries of the shapes
    shape_id = cmodel.getGeometryId(shape_name)
    shape = cmodel.geometryObjects[shape_id]
    Rot = R(rmodel, cmodel, q, shape_name=shape_name)
    shape_geom = shape.geometry  # Get the frame geom
    shape_radii = shape_geom.radii # Get the radii of shape 1
    D = np.array(
            [
                [1 / shape_radii[0] ** 2, 0, 0],
                [0, 1 / shape_radii[1] ** 2, 0],
                [0, 0, 1 / shape_radii[2] ** 2],
            ]
        )


    R_dot = dR_dt(rmodel, cmodel, x, shape_name=shape_name)

    A_dot = R_dot.T @ D @ Rot + Rot.T @ D @ R_dot
    return A_dot


def ddist_dt(rmodel, cmodel, x: np.ndarray, verbose = True):
    """Computing the derivative of the distance w.r.t. time.

    Args:
        rmodel (_type_): _description_
        cmodel (_type_): _description_
        x (np.ndarray): _description_
    """
    q = x[: rmodel.nq]
    v = x[rmodel.nq :]

    # Creating the data models
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Updating the position of the joints & the geometry objects.
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    pin.framesForwardKinematics(rmodel, rdata, q)

    pin.forwardKinematics(rmodel, rdata, q, v)
    # Poses and geometries of the shapes
    shape1_id = cmodel.getGeometryId("obstacle")
    shape1 = cmodel.geometryObjects[shape1_id]

    shape2_id = cmodel.getGeometryId("ellips_rob")
    shape2 = cmodel.geometryObjects[shape2_id]
    # Getting its pose in the world reference
    shape1_placement = cdata.oMg[shape1_id]
    # Doing the same for the second shape.
    shape2_geom = shape2.geometry
    shape2_placement = cdata.oMg[shape2_id]
    
    distance = dist(rmodel, cmodel, q)
    closest_points = cp(rmodel, cmodel, q)
    x1 = closest_points[:3]
    x2 = closest_points[3:]

    c1 = shape1_placement.translation
    c2 = shape2_placement.translation

    v1 = pin.getFrameVelocity(rmodel, rdata, shape1.parentFrame, pin.LOCAL_WORLD_ALIGNED)
    v2 = pin.getFrameVelocity(rmodel, rdata, shape2.parentFrame, pin.LOCAL_WORLD_ALIGNED)

    A_val = A(rmodel, cmodel,q)
    A1 = A_val[:3,:]
    A2 = A_val[3:,:]
    
    A_dot = dA_dt(rmodel, cmodel, x)
    A1_dot = A_dot[:3,:]
    A2_dot = A_dot[3:,:]

    n = (x2 - x1).T / distance
    vc1 = v1.linear
    vc2 = v2.linear
    d_dot = np.dot((vc2 - vc1 + (1/2) * np.linalg.pinv(A1) @  (x1 - c1) @ A1_dot - (1/2) * np.linalg.pinv(A2) @  (x2 - c2) @ A2_dot ),n)
    return d_dot

def ddist_dq(rmodel, cmodel, q):
    """Computing the derivative of the distance w.r.t. the configuration of the robot.

    Args:
        rmodel (_type_): _description_
        cmodel (_type_): _description_
        q (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Creating the data models
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Updating the position of the joints & the geometry objects.
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)

    # Poses and geometries of the shapes
    shape1_id = cmodel.getGeometryId("obstacle")
    shape1 = cmodel.geometryObjects[shape1_id]

    shape2_id = cmodel.getGeometryId("ellips_rob")
    shape2 = cmodel.geometryObjects[shape2_id]

    distance = dist(rmodel, cmodel, q)
    closest_points = cp(rmodel, cmodel, q)
    cp1 = closest_points[:3]
    cp2 = closest_points[3:]

    jacobian1 = pin.computeFrameJacobian(
        rmodel,
        rdata,
        q,
        shape1.parentFrame,
        pin.LOCAL_WORLD_ALIGNED,
    )

    jacobian2 = pin.computeFrameJacobian(
        rmodel,
        rdata,
        q,
        shape2.parentFrame,
        pin.LOCAL_WORLD_ALIGNED,
    )

    ## Transport the jacobian of frame 1 into the jacobian associated to cp1
    # Vector from frame 1 center to p1

    f1p1 = cp1 - rdata.oMf[shape1.parentFrame].translation
    # The following 2 lines are the easiest way to understand the transformation
    # although not the most efficient way to compute it.
    f1Mp1 = pin.SE3(np.eye(3), f1p1)
    jacobian1 = f1Mp1.actionInverse @ jacobian1

    ## Transport the jacobian of frame 2 into the jacobian associated to cp2
    # Vector from frame 2 center to p2
    f2p2 = cp2 - rdata.oMf[shape2.parentFrame].translation
    # The following 2 lines are the easiest way to understand the transformation
    # although not the most efficient way to compute it.
    f2Mp2 = pin.SE3(np.eye(3), f2p2)
    jacobian2 = f2Mp2.actionInverse @ jacobian2
    return (cp1 - cp2).T / distance @ (jacobian1[:3] - jacobian2[:3])



def numdiff_matrix(f, q, h=1e-6):
    fx = f(q).reshape(3,)
    j_diff = np.zeros((3, 7))
    for i in range(len(q)):
        e = np.zeros(len(q))
        e[i] = h
        j_diff[:, i] = (f(q + e).reshape(len(fx)) - fx) / e[i]
    return j_diff



def dX_dq(rmodel, cmodel, q):

    not_center = True
    derivativeComputation = DerivativeComputation()
    # Creating the data models
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Updating the position of the joints & the geometry objects.
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    pin.forwardKinematics(rmodel, rdata, q)
    # Poses and geometries of the shapes
    shape1_id = cmodel.getGeometryId("obstacle")
    shape1 = cmodel.geometryObjects[shape1_id]

    shape2_id = cmodel.getGeometryId("ellips_rob")
    shape2 = cmodel.geometryObjects[shape2_id]

    # Getting the geometry of the shape 1
    shape1_geom = shape1.geometry
    # Getting its pose in the world reference
    shape1_placement = cdata.oMg[shape1_id]
    # Doing the same for the second shape.
    shape2_geom = shape2.geometry
    shape2_placement = cdata.oMg[shape2_id]

    jacobian1 = pin.computeFrameJacobian(
        rmodel,
        rdata,
        q,
        shape1.parentFrame,
        pin.LOCAL_WORLD_ALIGNED,
    )

    jacobian2 = pin.computeFrameJacobian(
        rmodel,
        rdata,
        q,
        shape2.parentFrame,
        pin.LOCAL_WORLD_ALIGNED,
    )

    dx_dcenter = derivativeComputation.compute_dx_dcenter_hppfcl(
        shape1_geom, shape2_geom, shape1_placement, shape2_placement
    )
    closest_points = cp(rmodel, cmodel, q)
    cp1 = closest_points[:3].reshape((3,))
    cp2 = closest_points[3:].reshape((3,))

    if not_center:
        f1p1 = cp1 - rdata.oMf[shape1.parentFrame].translation
        f2p2 = cp2 - rdata.oMf[shape2.parentFrame].translation
        f1Mp1 = pin.SE3(np.eye(3), f1p1)
        jacobian1 = f1Mp1.actionInverse @ jacobian1
        f2Mp2 = pin.SE3(np.eye(3), f2p2)
        jacobian2 = f2Mp2.actionInverse @ jacobian2

    dx1_dcenter1 = dx_dcenter[:3, :3]
    dx1_dcenter2 = dx_dcenter[:3, 3:]
    dx2_dcenter1 = dx_dcenter[3:, :3]
    dx2_dcenter2 = dx_dcenter[3:, 3:]

    dcenter1_dq = jacobian1[:3]
    dcenter2_dq = jacobian2[:3]

    dx1_dq = (dcenter1_dq.T @ dx1_dcenter1).T + (dcenter2_dq.T @ dx1_dcenter2).T
    dx2_dq = (dcenter1_dq.T @ dx2_dcenter1).T + (dcenter2_dq.T @ dx2_dcenter2).T

    return np.concatenate((dx1_dq, dx2_dq))


def dddist_dt_dq(rmodel, cmodel, x):

    q = x[: rmodel.nq]
    v = x[rmodel.nq :]

    # Creating the data models
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Updating the position of the joints & the geometry objects.
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    pin.computeForwardKinematicsDerivatives(rmodel, rdata, q, v, a=np.zeros(rmodel.nq))
    # Poses and geometries of the shapes
    shape1_id = cmodel.getGeometryId("obstacle")
    shape1 = cmodel.geometryObjects[shape1_id]

    shape2_id = cmodel.getGeometryId("ellips_rob")
    shape2 = cmodel.geometryObjects[shape2_id]

    d = dist(rmodel, cmodel, q)
    closest_points = cp(rmodel, cmodel, q)
    cp1 = closest_points[:3]
    cp2 = closest_points[3:]

    f1p1 = cp1 - rdata.oMf[shape1.parentFrame].translation
    f2p2 = cp2 - rdata.oMf[shape2.parentFrame].translation

    dx_dq = dX_dq(rmodel, cmodel, q)
    dx1_dq = dx_dq[:3]
    dx2_dq = dx_dq[3:]

    dv1_dq = pin.getFrameVelocityDerivatives(
        rmodel, rdata, shape1.parentFrame, pin.LOCAL_WORLD_ALIGNED
    )[0][:3]
    dv2_dq = pin.getFrameVelocityDerivatives(
        rmodel, rdata, shape2.parentFrame, pin.LOCAL_WORLD_ALIGNED
    )[0][:3]

    v1 = pin.getFrameVelocity(
        rmodel, rdata, shape1.parentFrame, pin.LOCAL_WORLD_ALIGNED
    ).linear
    v2 = pin.getFrameVelocity(
        rmodel, rdata, shape2.parentFrame, pin.LOCAL_WORLD_ALIGNED
    ).linear

    v1 = np.reshape(v1, (3, 1))
    v2 = np.reshape(v2, (3, 1))

    return 1/d * np.reshape((dv1_dq - dv2_dq).T @ (f1p1 - f2p2), (rmodel.nq, 1)) + (dx1_dq - dx2_dq).T @ (v1 - v2)
    # return 1 / d * np.reshape((dv1_dq - dv2_dq).T @ (cp1 - cp2), (rmodel.nq, 1)) + (
    #     dx1_dq - dx2_dq
    # ).T @ (v1 - v2)


def h1(rmodel, cmodel, center, cp1,  q):
    
     # Creating the data models
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Updating the position of the joints & the geometry objects.
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    
     # Poses and geometries of the shapes
    shape1_id = cmodel.getGeometryId("obstacle")
    shape1 = cmodel.geometryObjects[shape1_id]

    shape2_id = cmodel.getGeometryId("ellips_rob")
    shape2 = cmodel.geometryObjects[shape2_id]
    
    R = rdata.oMf[shape1.parentFrame].rotation

    radii1 = shape1.geometry.radii
    D = np.diag([1 / r**2 for r in radii1])
    
    return (cp1 - center) @ R.T @ D @ R @ (cp1 - center) 

def h2(rmodel, cmodel,center, cp2,  q):
    
     # Creating the data models
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Updating the position of the joints & the geometry objects.
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    
     # Poses and geometries of the shapes
    shape2_id = cmodel.getGeometryId("ellips_rob")
    shape2 = cmodel.geometryObjects[shape2_id]
    
    R = rdata.oMf[shape2.parentFrame].rotation

    radii = shape2.geometry.radii
    D = np.diag([1 / r**2 for r in radii])
    
    return (cp2 - center) @ R.T @ D @ R @ (cp2 - center) 