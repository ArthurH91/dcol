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
        ksi = 1,
        di = 5e-2,
        ds = 1e-4
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
        
        self.ksi = ksi
        self.di = di
        self.ds = ds

    def f(self, x):
        ddist_dt_val = self.ddist_dt(self._pinocchio, self._geom_model, x)



        d = self.dist(self._pinocchio, self._geom_model, x)
        return ddist_dt_val + self.ksi * (d - self.ds) / (self.di - self.ds)

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
        
        ddistdot_dq_val = self.ddistdot_dq(self._pinocchio, self._geom_model, x)
        ddist_dq = np.r_[self.ddist_dq(self._pinocchio, self._geom_model, x), np.zeros(self._nq)]
        data.Rx[:] = ddistdot_dq_val - ddist_dq * self.ksi / (self.di - self.ds)
        
        # nd = numdiff(self.f, x)
        # print(np.linalg.norm(nd - data.Rx))
        # data.Rx[:] = nd

    def ddist_dq(self, rmodel, cmodel, x: np.ndarray):
        
        
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
        
        jacobian1 = pin.computeFrameJacobian(
            rmodel,
            rdata,
            q,
            self._shape1.parentFrame,
            pin.LOCAL_WORLD_ALIGNED,
        )

        jacobian2 = pin.computeFrameJacobian(
            rmodel,
            rdata,
            q,
            self._shape2.parentFrame,
            pin.LOCAL_WORLD_ALIGNED,
        )

        req = hppfcl.DistanceRequest()
        res = hppfcl.DistanceResult()
        # Computing the distance
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

        ## Transport the jacobian of frame 1 into the jacobian associated to x1
        # Vector from frame 1 center to p1
        f1p1 = x1 - rdata.oMf[self._shape1.parentFrame].translation
        # The following 2 lines are the easiest way to understand the transformation
        # although not the most efficient way to compute it.
        f1Mp1 = pin.SE3(np.eye(3), f1p1)
        jacobian1 = f1Mp1.actionInverse @ jacobian1

        ## Transport the jacobian of frame 2 into the jacobian associated to x2
        # Vector from frame 2 center to p2
        f2p2 = x2 - rdata.oMf[self._shape2.parentFrame].translation
        # The following 2 lines are the easiest way to understand the transformation
        # although not the most efficient way to compute it.
        f2Mp2 = pin.SE3(np.eye(3), f2p2)
        jacobian2 = f2Mp2.actionInverse @ jacobian2

        CP1_SE3 = pin.SE3.Identity()
        CP1_SE3.translation = x1

        CP2_SE3 = pin.SE3.Identity()
        CP2_SE3.translation = x2
        self._J = (x1 - x2).T / distance @ (jacobian1[:3] - jacobian2[:3])
        return self._J

    def ddistdot_dq(self, rmodel, cmodel, x: np.ndarray):
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

        D1 = np.diagflat(self._shape1.geometry.radii)
        D2 = np.diagflat(self._shape2.geometry.radii)
        
        R1 = shape1_placement.rotation
        R2 = shape2_placement.rotation
        A1, A2 = R1 @ D1 @ R1.T, R2 @ D2 @ R2.T  # From pinocchio A = RDR.T

        sol_lam1, sol_lam2 =  - (x1 - c1).T @ (x1 - x2) , (x2 - c2).T @ (x1 - x2) 

        theta_dot = np.r_[v1, v2, w1, w2]

        Ldot = (
            (x1 - x2) @ (v1 - v2)
            - np.cross(x1 - x2, x1 - c1) @ w1
            + np.cross(x1 - x2, x2 - c2) @ w2
        )
        dist_dot = Ldot / distance

        Lyy = np.r_[
            np.c_[np.eye(3) + sol_lam1 * A1, -np.eye(3), A1 @ (x1 - c1), np.zeros(3)],
            np.c_[-np.eye(3), np.eye(3) + sol_lam2 * A2, np.zeros(3), A2 @ (x2 - c2)],
            [np.r_[A1 @ (x1 - c1), np.zeros(3), np.zeros(2)]],
            [np.r_[np.zeros(3), A2 @ (x2 - c2), np.zeros(2)]],
        ]
        Lyc = np.r_[
            np.c_[-sol_lam1 * A1, np.zeros([3, 3])],
            np.c_[np.zeros([3, 3]), -sol_lam2 * A2],
            [np.r_[-A1 @ (x1 - c1), np.zeros(3)]],
            [np.r_[np.zeros(3), -A2 @ (x2 - c2)]],
        ]
        Lyr = np.r_[
            np.c_[
                sol_lam1 * (A1 @ pin.skew(x1 - c1) - pin.skew(A1 @ (x1 - c1))),
                np.zeros([3, 3]),
            ],
            np.c_[
                np.zeros([3, 3]),
                sol_lam2 * (A2 @ pin.skew(x2 - c2) - pin.skew(A2 @ (x2 - c2))),
            ],
            [np.r_[(x1 - c1) @ A1 @ pin.skew(x1 - c1), np.zeros(3)]],
            [np.r_[np.zeros(3), (x2 - c2) @ A2 @ pin.skew(x2 - c2)]],
        ]

        yc = -np.linalg.inv(Lyy) @ Lyc
        yr = -np.linalg.inv(Lyy) @ Lyr

        xc, xr = yc[:3], yr[:3]
        dx1 = np.c_[yc[:3], yr[:3]]
        dx2 = np.c_[yc[3:6], yr[3:6]]


        dL_dtheta = np.r_[
            x1 - x2,
            -(x1 - x2),
            -np.cross(x1 - x2, x1 - c1),
            np.cross(x1 - x2, x2 - c2),
        ]

        ddL_dtheta2 = (
            np.r_[
                dx1 - dx2,
                -dx1 + dx2,
                -pin.skew(x1 - x2) @ dx1 + pin.skew(x1 - c1) @ (dx1 - dx2),
                pin.skew(x1 - x2) @ dx2 - pin.skew(x2 - c2) @ (dx1 - dx2),
            ]
            + np.r_[
                np.zeros([6, 12]),
                np.c_[pin.skew(x1 - x2), np.zeros([3, 9])],
                np.c_[np.zeros([3, 3]), -pin.skew(x1 - x2), np.zeros([3, 6])],
            ]
        )

        d_dist_dot_dtheta = theta_dot.T @ ddL_dtheta2 / distance - dist_dot / distance**2 * dL_dtheta

        d_dist_dot_dtheta_dot = dL_dtheta / distance
        d_theta1_dq = pin.computeFrameJacobian(rmodel, rdata,q, self._shape1.parentFrame, pin.LOCAL_WORLD_ALIGNED)
        d_theta2_dq = pin.computeFrameJacobian(rmodel, rdata,q, self._shape2.parentFrame, pin.LOCAL_WORLD_ALIGNED)
        
        d_c1_dq = d_theta1_dq[:3]
        d_r1_dq = d_theta1_dq[3:]
        
        d_c2_dq = d_theta2_dq[:3]
        d_r2_dq = d_theta2_dq[3:]
        
        d_theta_dq = np.r_[d_c1_dq, d_c2_dq, d_r1_dq, d_r2_dq]
        
        d_theta_dot_dq = pin.computeJointJacobiansTimeVariation(rmodel, rdata, q, v)
        d_v1_dq = d_theta_dot_dq[:3,:]
        d_w1_dq = d_theta_dot_dq[3:,:]
        
        d_v2_dq = np.zeros((3, rmodel.nq))
        d_w2_dq = np.zeros((3, rmodel.nq))
        
        dJ = np.r_[d_v1_dq, d_v2_dq, d_w1_dq, d_w2_dq]
        d_dist_dot_dq = d_dist_dot_dtheta @ d_theta_dq + d_dist_dot_dtheta_dot @ dJ
        return np.r_[d_dist_dot_dq, d_dist_dot_dtheta_dot @ d_theta_dq]

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
