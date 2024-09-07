import numpy as np

import crocoddyl
import pinocchio as pin
import hppfcl
from distance_derivatives import dist


class ResidualModelVelocityAvoidance(crocoddyl.ResidualModelAbstract):
    """Class computing the residual of the collision constraint. This residual is simply the signed distance between the two closest points of the 2 shapes."""

    def __init__(
        self,
        state,
        geom_model: pin.Model,
        pair_id: int,
    ):
        """Class computing the residual of the collision constraint. This residual is simply the signed distance between the two closest points of the 2 shapes.

        Args:
            state (crocoddyl.StateMultibody): _description_
            geom_model (pin.Model): Collision model of pinocchio
            pair_id (int): ID of the collision pair
        """
        crocoddyl.ResidualModelAbstract.__init__(self, state, 1, True, True, True)

        # Pinocchio robot model
        self._pinocchio = self.state.pinocchio

        # Geometry model of the robot
        self._geom_model = geom_model

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

    def f(self, x):
        ddist_dt_val = self.ddist_dt(self._pinocchio, self._geom_model, x)

        di = 1e-2
        ds = 1e-5
        ksi = 0.01

        d = self.dist(self._pinocchio, self._geom_model, x)
        return ddist_dt_val + ksi * (d - ds) / (di - ds)

    def dist(self, rmodel, cmodel, x: np.ndarray):
        """Computing the distance between the two shapes.

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
        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)

        # Poses and geometries of the shapes
        shape1_placement = cdata.oMg[self._shape1_id]
        shape2_placement = cdata.oMg[self._shape2_id]

        req = hppfcl.DistanceRequest()
        res = hppfcl.DistanceResult()
        distance = hppfcl.distance(
            self._shape1.geometry,
            shape1_placement,
            self._shape2.geometry,
            shape2_placement,
            req,
            res,
        )
        return distance

    def ddist_dt(self, rmodel, cmodel, x: np.ndarray):
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
        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)

        # Poses and geometries of the shapes
        shape1_placement = cdata.oMg[self._shape1_id]
        shape2_placement = cdata.oMg[self._shape2_id]

        req = hppfcl.DistanceRequest()
        res = hppfcl.DistanceResult()
        distance = hppfcl.distance(
            self._shape1.geometry,
            shape1_placement,
            self._shape2.geometry,
            shape2_placement,
            req,
            res,
        )
        x1 = res.getNearestPoint1()
        x2 = res.getNearestPoint2()

        c1 = shape1_placement.translation
        c2 = shape2_placement.translation

        v1 = pin.getFrameVelocity(
            rmodel, rdata, self._shape1.parentFrame, pin.LOCAL_WORLD_ALIGNED
        ).linear
        v2 = pin.getFrameVelocity(
            rmodel, rdata, self._shape2.parentFrame, pin.LOCAL_WORLD_ALIGNED
        ).linear
        w1 = pin.getFrameVelocity(
            rmodel, rdata, self._shape1.parentFrame, pin.LOCAL_WORLD_ALIGNED
        ).angular
        w2 = pin.getFrameVelocity(
            rmodel, rdata, self._shape2.parentFrame, pin.LOCAL_WORLD_ALIGNED
        ).angular

        Lc = (x1 - x2).T
        Lr1 = c1.T @ pin.skew(x2 - x1) + x2.T @ pin.skew(x1)
        Lr2 = c2.T @ pin.skew(x1 - x2) + x1.T @ pin.skew(x2)

        Ldot = Lc @ (v1 - v2) + Lr1 @ w1 + Lr2 @ w2
        d_dot = Ldot / distance
        return d_dot

    def calc(self, data, x, u=None):
        data.r[:] = self.f(x)

    def calcDiff(self, data, x, u=None):
        nd = numdiff(self.f, x)
        data.Rx[:] = nd


def numdiff(f, q, h=1e-6):
    j_diff = np.zeros(len(q))
    fx = f(q)
    for i in range(len(q)):
        e = np.zeros(len(q))
        e[i] = h
        j_diff[i] = (f(q + e) - fx) / e[i]
    return j_diff


if __name__ == "__main__":
    pass
