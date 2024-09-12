import numpy as np

import crocoddyl
import pinocchio as pin
import hppfcl

from compute_deriv import compute_ddot, compute_d_d_dot_dq_dq_dot, compute_dist, compute_d_dist_dq

class ResidualModelVelocityAvoidance(crocoddyl.ResidualModelAbstract):
    """Class computing the residual of the collision constraint. This residual is simply the signed distance between the two closest points of the 2 shapes."""

    def __init__(
        self, state, geom_model: pin.Model,  idg1, idg2, ksi=1, di=5e-2, ds=1e-4
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


        self.idg1 = idg1
        self.idg2 = idg2
        
        self.ksi = ksi
        self.di = di
        self.ds = ds
   

    def f(self, x, u=None):
        
        d_dot = compute_ddot(self._pinocchio, self._geom_model, x[:7], x[7:], self.idg1,  self.idg2)        
        d = compute_dist(self._pinocchio, self._geom_model, x[:7], x[7:], self.idg1, self.idg2)        
        
        return d_dot + self.ksi * (d - self.ds) / (self.di - self.ds)

    def calc(self, data, x, u=None):
        data.r[:] = self.f(x, u)
        return data.r

    def calcDiff(self, data, x, u=None):

        ddistdot_dq, ddistdot_dq_dot = compute_d_d_dot_dq_dq_dot(self._pinocchio,  self._geom_model, x[:7], x[7:],  self.idg1,  self.idg2)
         
        # ddistdot_dq_nd = numdiff(
        #     lambda var: compute_ddot(self._pinocchio, self._geom_model, var[:7], var[7:], self.idg1,  self.idg2), x)
        # print(f"ddistdot_dq_val_test - ddistdot_dq_val_nd : {ddistdot_dq - ddistdot_dq_nd[:7]}")
        # print(f"x: {x}")
        
        # assert np.allclose(ddistdot_dq_val_nd[:7], ddistdot_dq_val_test, atol=1e-3), f"ddistdot_dq_val_nd: {ddistdot_dq_val_nd[:7]} != {ddistdot_dq_val_test}, {ddistdot_dq_val_nd[:7] - ddistdot_dq_val_test}, x: {x.tolist()}"

        ddist_dq = np.r_[
            compute_d_dist_dq(self._pinocchio, self._geom_model,x[:7], x[7:] , self.idg1,  self.idg2), np.zeros(self._pinocchio.nq)
        ]
        # ddist_dq_nd = numdiff(lambda var: compute_dist(self._pinocchio, self._geom_model, var[:7], var[7:], self.idg1, self.idg2), x)
        
        # assert np.allclose(ddist_dq[:7], ddist_dq_nd[:7], atol=1e-3), f"ddist_dq: {ddist_dq[:7]} != {ddist_dq_nd[:7]}, {ddist_dq[:7] - ddist_dq_nd[:7]}, x: {x.tolist()}"
        
        data.Rx[:] = np.r_[ddistdot_dq, ddistdot_dq_dot] - ddist_dq * self.ksi / (self.di - self.ds)
        # data.Rx[:] = ndf





def numdiff(f, q, h=1e-8):
    j_diff = np.zeros(len(q))
    fx = f(q)
    for i in range(len(q)):
        e = np.zeros(len(q))
        e[i] = h
        j_diff[i] = (f(q + e) - fx) / e[i]
    return j_diff

if __name__ == "__main__":
    pass
