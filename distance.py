import numpy as np
import pinocchio as pin


if __name__ == "__main__":
    from pinocchio import visualize
    import meshcat
    import meshcat.geometry as g
    from wrapper_panda import PandaWrapper

    # Creating the robot
    placement = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0, 2]))
    robot_wrapper = PandaWrapper()
    rmodel, cmodel, vmodel = robot_wrapper()

    # Creating the ellipsoids
    cmodel = robot_wrapper.add_ellipsoid(
        cmodel, "obstacle", placement=placement, dim=[0.1, 0.1, 0.4]
    )
    assert cmodel.existGeometryName("panda2_link7_sc_5")
    cmodel = robot_wrapper.add_ellipsoid(
        cmodel,
        "obstacle2",
        parentFrame=cmodel.geometryObjects[
            cmodel.getGeometryId("panda2_link7_sc_5")
        ].parentFrame,
        parentJoint=cmodel.geometryObjects[
            cmodel.getGeometryId("panda2_link7_sc_5")
        ].parentJoint,
        placement=pin.SE3(pin.utils.rotate("x", 2 * np.pi), np.array([0, 0, 0])),
        dim=[0.2, 0.1, 0.2],
    )

    # Creating the data models
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