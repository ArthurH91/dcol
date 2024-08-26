# 2-Clause BSD License

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from os.path import dirname, join, abspath

import numpy as np
import pinocchio as pin
import hppfcl


# This class is for unwrapping an URDF and converting it to a model. It is also possible to add objects in the model,
# such as a ball at a specific position.

RED = np.array([249, 136, 126, 125]) / 255


class PandaWrapper:
    def __init__(
        self,
    ):
        """Create a wrapper for the robot panda."""

        # Importing the model
        pinocchio_model_dir = dirname(((str(abspath(__file__)))))
        model_path = join(pinocchio_model_dir, "models")
        self._mesh_dir = join(model_path, "meshes")
        urdf_filename = "franka2.urdf"
        srdf_filename = "demo.srdf"
        self._urdf_model_path = join(join(model_path, "urdf"), urdf_filename)
        self._srdf_model_path = join(join(model_path, "srdf"), srdf_filename)

        # Color of the robot
        self._color = np.array([249, 136, 126, 255]) / 255

    def __call__(self):
        """Create a robot.

        Returns:
            rmodel (pin.Model): Model of the robot
            cmodel (pin.GeometryModel): Collision model of the robot
            vmodel (pin.GeometryModel): Visual model of the robot
        """
        (
            self._rmodel,
            self._cmodel,
            self._vmodel,
        ) = pin.buildModelsFromUrdf(
            self._urdf_model_path, self._mesh_dir, pin.JointModelFreeFlyer()
        )

        q0 = pin.neutral(self._rmodel)

        # Locking the gripper
        jointsToLockIDs = [1, 9, 10]

        geom_models = [self._vmodel, self._cmodel]
        self._model_reduced, geometric_models_reduced = pin.buildReducedModel(
            self._rmodel,
            list_of_geom_models=geom_models,
            list_of_joints_to_lock=jointsToLockIDs,
            reference_configuration=q0,
        )

        self._vmodel_reduced, self._cmodel_reduced = (
            geometric_models_reduced[0],
            geometric_models_reduced[1],
        )

        rdata = self._model_reduced.createData()
        cdata = self._cmodel_reduced.createData()
        q0 = pin.neutral(self._model_reduced)

        # Updating the models
        pin.framesForwardKinematics(self._model_reduced, rdata, q0)
        pin.updateGeometryPlacements(
            self._model_reduced, rdata, self._cmodel_reduced, cdata, q0
        )

        return (
            self._model_reduced,
            self._cmodel_reduced,
            self._vmodel_reduced,
        )

    def add_ellipsoid(
        self,
        cmodel,
        name: str,
        parentJoint=0,
        parentFrame=0,
        placement=pin.SE3.Random(),
        dim=[0.2, 0.5, 0.1],
    ):
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
    
    def add_2_ellips(
        self,
        cmodel,
        placement_obs=pin.SE3.Identity(),
        dim_obs=[1, 1, 1],
        placement_rob=pin.SE3.Identity(),
        dim_rob=[1, 1, 1],
    ):

        # Creating the ellipsoids
        cmodel = self.add_ellipsoid(
            cmodel, "obstacle", placement=placement_obs, dim=dim_obs
        )
        assert cmodel.existGeometryName("panda2_link7_sc_5")
        cmodel = self.add_ellipsoid(
            cmodel,
            "ellips_rob",
            parentFrame=cmodel.geometryObjects[
                cmodel.getGeometryId("panda2_link7_sc_0")
            ].parentFrame,
            parentJoint=cmodel.geometryObjects[
                cmodel.getGeometryId("panda2_link7_sc_0")
            ].parentJoint,
            placement=placement_rob,
            dim=dim_rob,
        )
        return cmodel


if __name__ == "__main__":
    from pinocchio import visualize
    import meshcat

    # Creating the robot
    placement = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0, 2]))
    robot_wrapper = PandaWrapper()
    rmodel, cmodel, vmodel = robot_wrapper()
    cmodel = robot_wrapper.add_elipsoid("test", placement=placement)
    rdata = rmodel.createData()
    cdata = cmodel.createData()

    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, pin.neutral(rmodel))
    # Generating the meshcat visualize
    #
    viz = visualize.MeshcatVisualizer(
        model=rmodel,
        collision_model=cmodel,
        visual_model=vmodel,
    )

    viz.initViewer(viewer=meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000"))
    viz.loadViewerModel("pinocchio")

    # vis[0].display(pin.randomConfiguration(rmodel))
    q = np.array(
        [
            1.06747267,
            1.44892299,
            -0.10145964,
            -2.42389347,
            2.60903241,
            3.45138352,
            -2.04166928,
        ]
    )
    viz.displayCollisions(True)
    viz.display(q)
