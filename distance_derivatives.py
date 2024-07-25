import numpy as np
import pinocchio as pin
import hppfcl

from derivatives_computation import DerivativeComputation


def dist(rmodel, cmodel, q):
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


def ddist_dq(rmodel, cmodel, q):

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


def ddist_dt(rmodel, cmodel, x: np.ndarray):

    q = x[: rmodel.nq]
    v = x[rmodel.nq :]

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

    n = (cp2 - cp1).T / distance

    d_dot = np.dot((jacobian2[:3] @ v - jacobian1[:3] @ v).T, n)

    return d_dot


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

    # return 1/d * ((dv1_dq - dv2_dq).T @ (f1p1 - f2p2), (rmodel.nq, 1)) + (dx1_dq - dx2_dq).T @ (v1 - v2)
    return 1 / d * np.reshape((dv1_dq - dv2_dq).T @ (cp1 - cp2), (rmodel.nq, 1)) + (
        dx1_dq - dx2_dq
    ).T @ (v1 - v2)
