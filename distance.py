import numpy as np
import pinocchio as pin


if __name__ == "__main__":
    from pinocchio import visualize
    import meshcat
    from wrapper_panda import PandaWrapper

    # Creating the robot
    placement = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 1, 2]))
    robot_wrapper = PandaWrapper()
    rmodel, cmodel, vmodel = robot_wrapper()
    for f in rmodel.joints:
        print(f)
    cmodel = robot_wrapper.add_ellipsoid(cmodel, "obstacle", placement=placement)
    cmodel = robot_wrapper.add_ellipsoid(
        cmodel,
        "obstacle2",
        parentFrame=cmodel.geometryObjects[
            cmodel.getGeometryId("panda2_link7_sc_5")
        ].parentFrame,
        parentJoint=cmodel.geometryObjects[
            cmodel.getGeometryId("panda2_link7_sc_5")
        ].parentJoint,
        placement=pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0, 4])),
        dim=[2,5,5]
    )

    print(cmodel)
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, pin.neutral(rmodel))
    # Generating the meshcat visualize
    viz = visualize.MeshcatVisualizer(
        model=rmodel,
        collision_model=cmodel,
        visual_model=vmodel,
    )
    viz.initViewer(
        # viewer=meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        )
    viz.loadViewerModel("pinocchio")

    q = pin.neutral(rmodel)
    viz.displayCollisions(True)
    viz.display(q)
    input()