import pinocchio as pin
import numpy as np
import hppfcl


class Scene:

    def __init__(self):
        pass

    def create_scene(self, cmodel: pin.GeometryModel, scene=1):
        """
        Creates a scene in the given `cmodel` based on the specified `scene` number.
        Parameters:
            cmodel (pin.GeometryModel): The geometry model to add the scene to.
            scene (int, optional): The scene number. Defaults to 1.
        Returns:
            pin.GeometryModel: The updated geometry model with the scene added.
        Raises:
            ValueError: If the specified `scene` is not implemented.
        """

        if scene == 1:

            # OBS CONSTANTS
            self.PLACEMENT_OBS = [pin.SE3(
                pin.utils.rotate("y", np.pi / 2) @ pin.utils.rotate("z", np.pi / 2),
                np.array([0, 0.1, 1.2]),
            )]
            self.DIM_OBS = [[0.1, 0.2, 0.1]]

            # ELLIPS ON THE ROBOT
            self.PLACEMENT_ROB = [pin.SE3(np.eye(3), np.array([0, 0, 0.0])), pin.SE3(pin.utils.rotate("z", np.pi) ,np.array([0, 0, 0.17]))]
            self.DIM_ROB =[ [0.1, 0.1, 0.17] , [0.04, 0.06, 0.04] ]
            
            # Adding the ellipsoids
            add_ellipsoid(
                cmodel, "obstacle", placement=self.PLACEMENT_OBS[0], dim=self.DIM_OBS[0]
            )
            add_ellipsoid(
                cmodel,
                "ellips_rob0",
                parentJoint=cmodel.geometryObjects[
                    cmodel.getGeometryId("panda2_link7_sc_5")
                ].parentJoint,
                parentFrame=cmodel.geometryObjects[
                    cmodel.getGeometryId("panda2_link7_sc_5")
                ].parentFrame,
                placement=self.PLACEMENT_ROB[0],
                dim=self.DIM_ROB[0],
            )

            
            add_ellipsoid(
                cmodel,
                "ellips_rob1",
                parentJoint=cmodel.geometryObjects[
                    cmodel.getGeometryId("panda2_leftfinger_0")
                ].parentJoint,
                parentFrame=cmodel.geometryObjects[
                    cmodel.getGeometryId("panda2_leftfinger_0")
                ].parentFrame,
                placement=self.PLACEMENT_ROB[1],
                dim=self.DIM_ROB[1],
            )


            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId("ellips_rob0"), cmodel.getGeometryId("obstacle")
                )
            )
            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId("ellips_rob1"), cmodel.getGeometryId("obstacle")
                )
            )
        elif scene == 2:

            # OBS CONSTANTS
            self.PLACEMENT_OBS = [
                pin.SE3(pin.utils.rotate("y", np.pi / 2),np.array([0.12, 0.2, 0.9])),
                pin.SE3(pin.utils.rotate("y", np.pi / 2),np.array([-0.12, 0.2, 0.9])),
                pin.SE3(pin.utils.rotate("y", np.pi / 2) @ pin.utils.rotate("x", np.pi / 2), np.array([-0.0, 0.05, 0.9])),
                pin.SE3(pin.utils.rotate("y", np.pi / 2) @ pin.utils.rotate("x", np.pi / 2), np.array([-0.0, 0.35, 0.9])),


                ]
            self.DIM_OBS = [[0.09, 0.15, 0.03], [0.09, 0.15, 0.03], [0.09, 0.15, 0.03], [0.09, 0.15, 0.03]]

            # ELLIPS ON THE ROBOT
            self.PLACEMENT_ROB = [pin.SE3(np.eye(3), np.array([0, 0, 0.0])), pin.SE3(pin.utils.rotate("z", np.pi) ,np.array([0, 0, 0.17]))]
            self.DIM_ROB =[ [0.1, 0.1, 0.17] , [0.04, 0.06, 0.04] ]
            
            # Adding the ellipsoids
            add_ellipsoid(
                cmodel, "obstacle0", placement=self.PLACEMENT_OBS[0], dim=self.DIM_OBS[0]
            )
            
            add_ellipsoid(
                cmodel, "obstacle1", placement=self.PLACEMENT_OBS[1], dim=self.DIM_OBS[1]
            )
            
            add_ellipsoid(
                cmodel, "obstacle2", placement=self.PLACEMENT_OBS[2], dim=self.DIM_OBS[2]
            )
            
            add_ellipsoid(
                cmodel, "obstacle3", placement=self.PLACEMENT_OBS[3], dim=self.DIM_OBS[3]
            )
            
            
            
            add_ellipsoid(
                cmodel,
                "ellips_rob0",
                parentJoint=cmodel.geometryObjects[
                    cmodel.getGeometryId("panda2_link7_sc_5")
                ].parentJoint,
                parentFrame=cmodel.geometryObjects[
                    cmodel.getGeometryId("panda2_link7_sc_5")
                ].parentFrame,
                placement=self.PLACEMENT_ROB[0],
                dim=self.DIM_ROB[0],
            )

            
            add_ellipsoid(
                cmodel,
                "ellips_rob1",
                parentJoint=cmodel.geometryObjects[
                    cmodel.getGeometryId("panda2_leftfinger_0")
                ].parentJoint,
                parentFrame=cmodel.geometryObjects[
                    cmodel.getGeometryId("panda2_leftfinger_0")
                ].parentFrame,
                placement=self.PLACEMENT_ROB[1],
                dim=self.DIM_ROB[1],
            )


            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId("ellips_rob0"), cmodel.getGeometryId("obstacle0")
                )
            )
            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId("ellips_rob1"), cmodel.getGeometryId("obstacle0")
                )
            )
            
            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId("ellips_rob0"), cmodel.getGeometryId("obstacle1")
                )
            )
            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId("ellips_rob1"), cmodel.getGeometryId("obstacle1")
                )
            )
            
            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId("ellips_rob0"), cmodel.getGeometryId("obstacle2")
                )
            )
            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId("ellips_rob1"), cmodel.getGeometryId("obstacle2")
                )
            )
            
            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId("ellips_rob0"), cmodel.getGeometryId("obstacle3")
                )
            )
            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId("ellips_rob1"), cmodel.getGeometryId("obstacle3")
                )
            )
            
            
        else:
            raise ValueError("Scene not implemented")
        return cmodel

    def get_initial_config(self, scene=1):
        """
        Get the initial configuration of the robot based on the specified `scene` number.
        Parameters:
            scene (int, optional): The scene number. Defaults to 1.
        Returns:
            np.ndarray: The initial configuration of the robot.
        Raises:
            ValueError: If the specified `scene` is not implemented.
        """

        if scene == 1:
            return np.array(
                [
                    -0.06709294,
                    1.35980773,
                    -0.81605989,
                    0.74243348,
                    0.42419277,
                    0.45547585,
                    -0.00456262,
                ]
            )
        elif scene == 2:
            return np.array(
                [
                    -0.06709294,
                    1.35980773,
                    -0.81605989,
                    0.74243348,
                    0.42419277,
                    0.45547585,
                    -0.00456262,
                ]
            )
        else:
            raise ValueError("Scene not implemented")
        
    def get_target_pose(self, scene=1):
        """
        Get the target pose based on the specified `scene` number.
        Parameters:
            scene (int, optional): The scene number. Defaults to 1.
        Returns:
            pin.SE3: The target pose.
        Raises:
            ValueError: If the specified `scene` is not implemented.
        """

        if scene == 1:
            return pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0.5, 1.2]))
        elif scene == 2:
            return pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0.2, 0.9]))
        else:
            raise ValueError("Scene not implemented")

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
