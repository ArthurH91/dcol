import numpy as np
import pinocchio as pin
import hppfcl


def add_ellips(
    cmodel,
    placement_obs=pin.SE3.Identity(),
    dim_obs=[1, 1, 1],
    placement_rob=pin.SE3.Identity(),
    dim_rob=[1, 1, 1],
):

    # Creating the ellipsoids
    cmodel = robot_wrapper.add_ellipsoid(
        cmodel, "obstacle", placement=placement_obs, dim=dim_obs
    )
    assert cmodel.existGeometryName("panda2_link7_sc_5")
    cmodel = robot_wrapper.add_ellipsoid(
        cmodel,
        "ellips_rob",
        parentFrame=cmodel.geometryObjects[
            cmodel.getGeometryId("panda2_link7_sc_5")
        ].parentFrame,
        parentJoint=cmodel.geometryObjects[
            cmodel.getGeometryId("panda2_link7_sc_5")
        ].parentJoint,
        placement=placement_rob,
        dim=dim_rob,
    )
    return cmodel


def add_closest_points(cmodel, rmodel, q):

    # Creating the data models
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Updating the geometry placements
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)

    req = hppfcl.DistanceRequest()
    res = hppfcl.DistanceResult()
    _ = hppfcl.distance(
        cmodel.geometryObjects[cmodel.getGeometryId("obstacle")].geometry,
        cdata.oMg[cmodel.getGeometryId("obstacle")],
        cmodel.geometryObjects[cmodel.getGeometryId("ellips_rob")].geometry,
        cdata.oMg[cmodel.getGeometryId("ellips_rob")],
        req,
        res,
    )
    cp1 = res.getNearestPoint1()
    cp2 = res.getNearestPoint2()
    return cp1, cp2


def dd_dt(rmodel, cmodel, x: np.ndarray):

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

    cp1 = res.getNearestPoint1()
    cp2 = res.getNearestPoint2()

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

    CP1_SE3 = pin.SE3.Identity()
    CP1_SE3.translation = cp1

    CP2_SE3 = pin.SE3.Identity()
    CP2_SE3.translation = cp2

    n = (cp2 - cp1).T / dist

    d_dot = np.dot((jacobian2[:3] @ v - jacobian1[:3] @ v).T, n)

    return d_dot
    # h = ksi * (dist - ds)/(di - ds)


def dist(q):
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


def finite_diff_time(q, v, h=1e-6):

    return (dist(q + h * v) - dist(q)) / h


def numdiff(f, q, h=1e-6):
    j_diff = np.zeros(rmodel.nq)
    fx = f(q)
    for i in range(rmodel.nq):
        e = np.zeros(rmodel.nq)
        e[i] = h
        j_diff[i] = (f(q + e) - fx) / e[i]
    return j_diff


if __name__ == "__main__":
    from wrapper_panda import PandaWrapper
    from viewer import create_viewer, display_closest_points

    # OBS CONSTANTS
    PLACEMENT_OBS = pin.SE3(pin.utils.rotate("x", 0), np.array([0, 0, 2]))
    DIM_OBS = [0.1, 0.1, 0.4]

    # ELLIPS ON THE ROBOT
    PLACEMENT_ROB = pin.SE3(pin.utils.rotate("x", 0), np.array([0, 0, 0]))
    DIM_ROB = [0.2, 0.1, 0.2]

    # Creating the robot
    robot_wrapper = PandaWrapper()
    rmodel, cmodel, vmodel = robot_wrapper()

    cmodel = add_ellips(
        cmodel,
        placement_obs=PLACEMENT_OBS,
        dim_obs=DIM_OBS,
        placement_rob=PLACEMENT_ROB,
        dim_rob=DIM_ROB,
    )
    viz = create_viewer(rmodel, cmodel, vmodel)
    q = pin.randomConfiguration(rmodel)
    v = pin.randomConfiguration(rmodel)
    cp1, cp2 = add_closest_points(cmodel, rmodel, q)
    display_closest_points(viz, cp1, cp2)
    # Generating the meshcat visualize
    viz.display(q)

    print(dd_dt(rmodel, cmodel, np.concatenate((q, v))))
    print(finite_diff_time(q, v))
