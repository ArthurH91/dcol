import numpy as np
import pinocchio as pin


if __name__ == "__main__":
    import hppfcl
    from pinocchio import visualize
    import meshcat
    import meshcat.geometry as g
    from wrapper_panda import PandaWrapper

    # OBS CONSTANTS 
    PLACEMENT_OBS = pin.SE3(pin.utils.rotate("x", 0), np.array([0, 0, 2]))
    DIM_OBS = [0.1, 0.1, 0.4]
    
    # ELLIPS ON THE ROBOT
    PLACEMENT_ROB= pin.SE3(pin.utils.rotate("x", 0), np.array([0, 0, 0]))
    DIM_ROB= [0.2, 0.1, 0.2]
    
    # Creating the robot
    robot_wrapper = PandaWrapper()
    rmodel, cmodel, vmodel = robot_wrapper()

    # Creating the ellipsoids
    cmodel = robot_wrapper.add_ellipsoid(
        cmodel, "obstacle", placement=PLACEMENT_OBS, dim=DIM_OBS
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
        placement=PLACEMENT_ROB,
        dim=DIM_ROB,
    )
    # Creating the data models
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Updating the geometry placements
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, pin.neutral(rmodel))

    req = hppfcl.DistanceRequest()
    res = hppfcl.DistanceResult()
    dist = hppfcl.distance(
        cmodel.geometryObjects[
            cmodel.getGeometryId("obstacle")].geometry, 
        cdata.oMg[cmodel.getGeometryId("obstacle")] ,
        cmodel.geometryObjects[
            cmodel.getGeometryId("ellips_rob")].geometry, 
        cdata.oMg[cmodel.getGeometryId("ellips_rob")],
        req,
        res
    )
    cp1 = res.getNearestPoint1()
    placement_cp1 = pin.SE3(np.eye(3), cp1)

    cp2 = res.getNearestPoint2()
    placement_cp2 = pin.SE3(np.eye(3), cp2)
    print(f"cp1: {cp1}")
    print(f"cp2: {cp2}")
    print(f"dist : {dist}")
    print(f"np.linalg.norm(cp2-cp1): {np.linalg.norm(cp2-cp1)}")
    print("-----------------")
    geom_cp = hppfcl.Sphere(0.05)
    cp1_geom = pin.GeometryObject("cp1", 0, 0, placement_cp1, geom_cp)
    cp2_geom = pin.GeometryObject("cp2", 0, 0, placement_cp2, geom_cp)
    cp1_geom.meshColor = np.array([0, 0, 0, 1.])
    cp2_geom.meshColor = np.array([0, 0, 0, 1.])
    cmodel.addGeometryObject(cp1_geom)
    cmodel.addGeometryObject(cp2_geom)

    rdata = rmodel.createData()
    cdata = cmodel.createData()

    # Updating the geometry placements
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, pin.neutral(rmodel))

    # Generating the meshcat visualize
    viz = visualize.MeshcatVisualizer(
        model=rmodel,
        collision_model=cmodel,
        visual_model=vmodel,
    )
    viz.initViewer(viewer=meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000"))
    viz.loadViewerModel("pinocchio")

    q = pin.neutral(rmodel)
    viz.displayCollisions(True)

    viz.display(q)