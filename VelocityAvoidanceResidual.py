import crocoddyl
import pinocchio as pin

from distance_derivatives import ddist_dt, dddist_dt_dq, dist, ddist_dq

class ResidualModelVelocityAvoidance(crocoddyl.ResidualModelAbstract):
    """Class computing the residual of the collision constraint. This residual is simply the signed distance between the two closest points of the 2 shapes."""

    def __init__(
        self,
        state,
        geom_model: pin.Model,
        geom_data,
        pair_id: int,
    ):
        """Class computing the residual of the collision constraint. This residual is simply the signed distance between the two closest points of the 2 shapes.

        Args:
            state (crocoddyl.StateMultibody): _description_
            geom_model (pin.Model): Collision model of pinocchio
            geom_data (_type_): Collision data of the collision model of pinocchio
            pair_id (int): ID of the collision pair
        """
        crocoddyl.ResidualModelAbstract.__init__(self, state, 1, True, True, True)

        # Pinocchio robot model
        self._pinocchio = self.state.pinocchio

        # Geometry model of the robot
        self._geom_model = geom_model

        # Geometry data of the robot
        self._geom_data = geom_data

        # Pair ID of the collisionPair
        self._pair_id = pair_id

        # Number of joints
        self._nq = self._pinocchio.nq

        # Making sure that the pair of collision exists
        assert self._pair_id <= len(self._geom_model.collisionPairs)

        # Collision pair
        self._collisionPair = self._geom_model.collisionPairs[self._pair_id]

        # Geometry ID of the shape 1 of collision pair
        self._shape1_id = self._collisionPair.first

        # Making sure that the frame exists
        assert self._shape1_id <= len(self._geom_model.geometryObjects)

        # Geometry object shape 1
        self._shape1 = self._geom_model.geometryObjects[self._shape1_id]

        # Shape 1 parent joint
        self._shape1_parentJoint = self._shape1.parentJoint

        # Geometry ID of the shape 2 of collision pair
        self._shape2_id = self._collisionPair.second

        # Making sure that the frame exists
        assert self._shape2_id <= len(self._geom_model.geometryObjects)

        # Geometry object shape 2
        self._shape2 = self._geom_model.geometryObjects[self._shape2_id]

        # Shape 2 parent joint
        self._shape2_parentJoint = self._shape2.parentJoint
        
        self.alpha = 10

    def calc(self, data, x, u=None):
        data.r[:] = ddist_dt(self._pinocchio, self._geom_model, x) - self.alpha * dist(self._pinocchio, self._geom_model, x[:self._pinocchio.nq])

    def calcDiff(self, data, x, u=None):
        data.Rx[: self._nq] = dddist_dt_dq(self._pinocchio, self._geom_model, x) - self.alpha * ddist_dq(self._pinocchio, self._geom_model, x[:self._pinocchio.nq])

if __name__ == "__main__":
    pass