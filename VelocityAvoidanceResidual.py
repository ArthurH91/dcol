import numpy as np

import crocoddyl
import pinocchio as pin

from distance_derivatives import ddist_dt, dddist_dt_dq, dist, ddist_dq

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

    def numdiff_ddist_dt(self, x):
        q = x[:self._nq]
        v = x[self._nq:]
        
        d = lambda config: dist(self._pinocchio, self._geom_model,config)
        return finite_diff_time(q, v, d)
    
    def f(self, x):
        ddist_dt_val = self.numdiff_ddist_dt(x)
        
        di = 1e-1
        ds = 1e-3
        ksi = 1
        
        d = dist(self._pinocchio, self._geom_model, x[:self._nq])
        # print(f"d : {d}")
        # print(f"ctr : {ddist_dt_val + ksi * (d - ds)/(di-ds)}")
        return ddist_dt_val + ksi * (d - ds)/(di-ds)
    def calc(self, data, x, u=None):
        data.r[:] = self.f(x)
        
    def calcDiff(self, data, x, u=None):
        nd = numdiff(self.f, x)
        
        # print(f"nd : {nd}")
        data.Rx[:] = nd
        
def finite_diff_time(q, v, f, h=1e-6):
    return (f(q + h * v) - f(q)) / h

def numdiff(f, q, h=1e-4):
    j_diff = np.zeros(len(q))
    fx = f(q)
    for i in range(len(q)):
        e = np.zeros(len(q))
        e[i] = h
        j_diff[i] = (f(q + e) - fx) / e[i]
    return j_diff
if __name__ == "__main__":
    pass