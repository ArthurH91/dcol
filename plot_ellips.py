import numpy as np
import pinocchio as pin
import hppfcl

from qcqp_solver import radii_to_matrix, EllipsoidOptimization


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
    d = hppfcl.distance(
        cmodel.geometryObjects[cmodel.getGeometryId("obstacle")].geometry,
        cdata.oMg[cmodel.getGeometryId("obstacle")],
        cmodel.geometryObjects[cmodel.getGeometryId("ellips_rob")].geometry,
        cdata.oMg[cmodel.getGeometryId("ellips_rob")],
        req,
        res,
    )
    cp1 = res.getNearestPoint1()
    cp2 = res.getNearestPoint2()
    return cp1, cp2,d



if __name__ == "__main__":
    from wrapper_panda import PandaWrapper
    from viewer import create_viewer, display_closest_points
    import casadi as ca
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
    q = pin.randomConfiguration(rmodel)
    v = pin.randomConfiguration(rmodel)
    cp1, cp2,d = add_closest_points(cmodel, rmodel, q)
    viz = create_viewer(rmodel, cmodel, vmodel)



    display_closest_points(viz, cp1, cp2, "cp1_0", "cp2_0")
    
    
    
    A = np.diag([1 / r**2 for r in DIM_OBS])
    B = np.diag([1 / r**2 for r in DIM_ROB])
    
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Updating the geometry placements
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)

    R_A = cdata.oMg[cmodel.getGeometryId("obstacle")].rotation
    R_B = cdata.oMg[cmodel.getGeometryId("ellips_rob")].rotation
    
    x0_1 = cdata.oMg[cmodel.getGeometryId("obstacle")].translation
    x0_2 = cdata.oMg[cmodel.getGeometryId("ellips_rob")].translation
    # Calculate rotated matrices
    A_rot = R_A.T @ A @ R_A
    B_rot = R_B.T @ B @ R_B

    # Setup optimization problem with CasADi
    opti = ca.Opti()
    x1 = opti.variable(3)
    x2 = opti.variable(3)

    # Define the cost function (distance between points)
    totalcost = ca.norm_2(x1 - x2)

    # Define constraints for the ellipsoids
    con1 = (x1 - x0_1).T @ A_rot @ (x1 - x0_1) <= 1
    con2 = (x2 - x0_2).T @ B_rot @ (x2 - x0_2) <= 1
    opti.subject_to([con1, con2])

    opti.solver('ipopt')
    opti.minimize(totalcost)

    # Apply warm start values
    opti.set_initial(x1, x0_1)
    opti.set_initial(x2, x0_2)

    # Solve the optimization problem
    try:
        solution = opti.solve()
        x1_sol = opti.value(x1)
        x2_sol = opti.value(x2)
    except RuntimeError as e:
        print(f"Solver failed: {e}")
        x1_sol = opti.debug.value(x1)
        x2_sol = opti.debug.value(x2)
        print("Debug values:")
        print("x1:", x1_sol)
        print("x2:", x2_sol)
        print("Total cost:", opti.debug.value(totalcost))
        raise

    # Print results for comparison
    print(f"x1 CasADi: {x1_sol} || x1 HPP-FCL: {cp1} || x1 DIFF: {x1_sol - cp1}")
    print(f"x2 CasADi: {x2_sol} || x2 HPP-FCL: {cp2} || x2 DIFF: {x2_sol - cp2}")
    print(f"Distance CasAdi: {opti.debug.value(totalcost)} || Distance HPP-FCL {d} || DIFF {opti.debug.value(totalcost) - d}")

    # Generating the meshcat visualize
    viz.display(q)
