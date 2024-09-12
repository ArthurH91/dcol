import numpy as np

import crocoddyl
import pinocchio as pin
import hppfcl

from compute_deriv import compute_ddot, compute_d_d_dot_dq_dq_dot

class ResidualModelVelocityAvoidance(crocoddyl.ResidualModelAbstract):
    """Class computing the residual of the collision constraint. This residual is simply the signed distance between the two closest points of the 2 shapes."""

    def __init__(
        self, state, nu, geom_model: pin.Model, pair_id: int, ksi=1, di=5e-2, ds=1e-4
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
        self._shape1_parentFrame = self._shape1.parentFrame

        # Geometry ID of the shape 2 of collision pair
        self._shape2_id = self._collisionPair.second

        # Making sure that the frame exists
        assert self._shape2_id <= len(self._geom_model.geometryObjects)

        # Geometry object shape 2
        self._shape2 = self._geom_model.geometryObjects[self._shape2_id]

        # Shape 2 parent joint
        self._shape2_parentJoint = self._shape2.parentJoint
        self._shape2_parentFrame = self._shape2.parentFrame

        self.ksi = ksi
        self.di = di
        self.ds = ds
   

    def f(self, x, u=None):
        
        d_dot_val = d_dot(self._pinocchio, self._geom_model, x, self._shape1_id, self._shape2_id)
        d_dot_from_n = compute_ddot(self._pinocchio, self._geom_model, x[:7], x[7:], self._shape1_id, self._shape2_id)
        
        assert np.allclose(d_dot_from_n, d_dot_val, atol=1e-8), f"ddot: {d_dot_from_n} != {d_dot_val}, x: {x}"
        
        d = dist(self._pinocchio, self._geom_model, x, self._shape1_id, self._shape2_id)
        
        return d_dot_val + self.ksi * (d - self.ds) / (self.di - self.ds)

    def calc(self, data, x, u=None):
        data.r[:] = self.f(x, u)
        return data.r

    def calcDiff(self, data, x, u=None):

        
        ddistdot_dq_val,ddistdot_dq_dot_val  = d_ddot_dq(self._pinocchio, self._geom_model, x, self._shape1_id, self._shape2_id)
        ddistdot_dq_val_test, ddistdot_dq_dot_val_test = compute_d_d_dot_dq_dq_dot(self._pinocchio, self._geom_model, x[:7], x[7:], self._shape1_id, self._shape2_id)
         
        ddistdot_dq_val_nd = numdiff(
            lambda var: d_dot(self._pinocchio, self._geom_model, var, self._shape1_id, self._shape2_id), x)
        # print(f"ddistdot_dq_val_test - ddistdot_dq_val_nd : {ddistdot_dq_val_test - ddistdot_dq_val_nd[:7]}")
        # print(f"x: {x}")
        
        # assert np.allclose(ddistdot_dq_val_nd[:7], ddistdot_dq_val_test, atol=1e-1), f"""ddistdot_dq_val_nd: {ddistdot_dq_val_nd[:7]} != {ddistdot_dq_val_test}, {ddistdot_dq_val_nd[:7] - ddistdot_dq_val_test}, x: {x.tolist()}
        # ddistdot_dq_val_test: {ddistdot_dq_val_test }, ddistdot_dq_val_nd[:7]: {ddistdot_dq_val_nd[:7]}"""

        ddist_dq = np.r_[
            d_dist_dq(self._pinocchio, self._geom_model, x, self._shape1_id, self._shape2_id), np.zeros(self._nq)
        ]
        data.Rx[:] = np.r_[ddistdot_dq_val, ddistdot_dq_dot_val] - ddist_dq * self.ksi / (self.di - self.ds)
        # ndf = numdiff(self.f, x)
        # print(f"Rx: {data.Rx}")
        # print(f"ndf: {ndf}")
        # print(f"diff Rq: {(data.Rx - ndf)[:7]}")
        # print(f"diff Rvq: {(data.Rx - ndf)[7:]}")
        # print(f"diff ddistdot_dq_val_nd: {ddistdot_dq_val_nd}")
        # print(f"diff ddistdot_dq_val: {ddistdot_dq_val}")
        # print(f"diff ddistdot_dq_val - ddistdot_dq_val_nd: {ddistdot_dq_val - ddistdot_dq_val_nd}")
        # print(f"x = {x}")
        # print("________")
        # print(f"no nd: {np.linalg.norm(ddistdot_dq_val)}")
        # print(f"nd : {np.linalg.norm(nd)}")
        # print(nd -ddistdot_dq_val)
        # print("---------")
        # data.Rx[:] = ndf



def dist(rmodel, cmodel, x: np.ndarray, shape1_id: int, shape2_id: int):
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
        shape1_placement = cdata.oMg[shape1_id]
        shape2_placement = cdata.oMg[shape2_id]
        shape1 = cmodel.geometryObjects[shape1_id]
        shape2 = cmodel.geometryObjects[shape2_id]


        req = hppfcl.DistanceRequest()
        # req.gjk_max_iterations = 20000
        # req.abs_err = 0
        # req.gjk_tolerance = 1e-9
        res = hppfcl.DistanceResult()
        
        distance = hppfcl.distance(
            shape1.geometry,
            shape1_placement,
            shape2.geometry,
            shape2_placement,
            req,
            res,
        )
        return distance

def d_dot(rmodel, cmodel, x: np.ndarray, shape1_id: int, shape2_id: int):
        """Computing the derivative of the distance w.r.t. time.

        Args:
            rmodel (_type_): _description_
            cmodel (_type_): _description_
            x (np.ndarray): _description_
        """

        q = x[: rmodel.nq]
        vq = x[rmodel.nq :]

        # Creating the data models
        rdata = rmodel.createData()
        cdata = cmodel.createData()

        # Updating the position of the joints & the geometry objects.
        pin.forwardKinematics(rmodel, rdata, q, vq)
        pin.updateFramePlacements(rmodel, rdata)
        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)

        # Poses and geometries of the shapes
        shape1_placement = cdata.oMg[shape1_id]
        shape2_placement = cdata.oMg[shape2_id]

        shape1 = cmodel.geometryObjects[shape1_id]
        shape2 = cmodel.geometryObjects[shape2_id]
        
        req = hppfcl.DistanceRequest()
        req.gjk_max_iterations = 20000
        req.abs_err = 0
        req.gjk_tolerance = 1e-9
        res = hppfcl.DistanceResult()
        
        distance = hppfcl.distance(
            shape1.geometry,
            shape1_placement,
            shape2.geometry,
            shape2_placement,
            req,
            res,
        )
        x1 = res.getNearestPoint1()
        x2 = res.getNearestPoint2()

        c1 = shape1_placement.translation
        c2 = shape2_placement.translation

        v1 = pin.getFrameVelocity(
            rmodel, rdata, shape1.parentFrame, pin.LOCAL_WORLD_ALIGNED
        ).linear
        v2 = pin.getFrameVelocity(
            rmodel, rdata, shape2.parentFrame, pin.LOCAL_WORLD_ALIGNED
        ).linear
        w1 = pin.getFrameVelocity(
            rmodel, rdata, shape1.parentFrame, pin.LOCAL_WORLD_ALIGNED
        ).angular
        w2 = pin.getFrameVelocity(
            rmodel, rdata, shape2.parentFrame, pin.LOCAL_WORLD_ALIGNED
        ).angular

        # Witness point velocities (seen as fixed on the objects)
        vx1 = v1 + np.cross(w1,x1-c1)
        vx2 = v2 + np.cross(w2,x2-c2)

        # Cost time derivative
        Ldot = (x1-x2)@(vx1-vx2)
        # Distance time derivative
        dist_dot = Ldot/distance
        return dist_dot
    
        rdata = rmodel.createData()
        gdata = cmodel.createData()
        # Compute robot placements and velocity at step 0 and 1
        pin.forwardKinematics(rmodel,rdata,q,vq)
        pin.updateFramePlacements(rmodel,rdata)
        pin.updateGeometryPlacements(rmodel,rdata,cmodel,gdata)

        elips1 = cmodel.geometryObjects[idg1].geometry
        elips2 = cmodel.geometryObjects[idg2].geometry

        idf1 = cmodel.geometryObjects[idg1].parentFrame
        idf2 = cmodel.geometryObjects[idg2].parentFrame
        
        M1 = gdata.oMg[idg1].copy()
        c1 = M1.translation
        M2 = gdata.oMg[idg2].copy()
        c2 = M2.translation

        # Get body velocity at step 0
        nu1 = pin.getFrameVelocity(rmodel,rdata,idf1,pin.LOCAL_WORLD_ALIGNED).copy()
        v1,w1 = nu1.linear,nu1.angular
        nu2 = pin.getFrameVelocity(rmodel,rdata,idf2,pin.LOCAL_WORLD_ALIGNED).copy()
        v2,w2 = nu2.linear,nu2.angular

        req = hppfcl.DistanceRequest()
        req.gjk_max_iterations = 20000
        req.abs_err = 0
        req.gjk_tolerance = 1e-9
        res = hppfcl.DistanceResult()
        distance = hppfcl.distance(
            elips1,
            gdata.oMg[idg1],
            elips2,
            gdata.oMg[idg2],
            req,
            res,
        )
        sol_x1 = res.getNearestPoint1()
        sol_x2 = res.getNearestPoint2()

        Ldot = (sol_x1-sol_x2)@(v1-v2) \
            - np.cross(sol_x1-sol_x2,sol_x1-c1)@w1 \
            + np.cross(sol_x1-sol_x2,sol_x2-c2)@w2

        return Ldot/distance

    
def d_dist_dq(rmodel, cmodel, x: np.ndarray, shape1_id: int, shape2_id: int):

        q = x[: rmodel.nq]
        v = x[rmodel.nq :]

        # Creating the data models
        rdata = rmodel.createData()
        cdata = cmodel.createData()
        # Updating the position of the joints & the geometry objects.
        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)

        # Poses and geometries of the shapes
        shape1_placement = cdata.oMg[shape1_id]
        shape2_placement = cdata.oMg[shape2_id]
        
        shape1 = cmodel.geometryObjects[shape1_id]
        shape2 = cmodel.geometryObjects[shape2_id]

        jacobian1 = pin.computeFrameJacobian(
            rmodel,
            rdata,
            q,
            shape1.parentFrame,
            pin.LOCAL_WORLD_ALIGNED,
        )

        jacobian2 = pin.computeFrameJacobian(
            rmodel,
            rdata,
            q,
            shape2.parentFrame,
            pin.LOCAL_WORLD_ALIGNED,
        )
        req = hppfcl.DistanceRequest()
        req.gjk_max_iterations = 20000
        req.abs_err = 0
        req.gjk_tolerance = 1e-9
        res = hppfcl.DistanceResult()
        # Computing the distance
        distance = hppfcl.distance(
            shape1.geometry,
            shape1_placement,
            shape2.geometry,
            shape2_placement,
            req,
            res,
        )
        x1 = res.getNearestPoint1()
        x2 = res.getNearestPoint2()

        ## Transport the jacobian of frame 1 into the jacobian associated to x1
        # Vector from frame 1 center to p1
        f1p1 = x1 - rdata.oMf[shape1.parentFrame].translation
        # The following 2 lines are the easiest way to understand the transformation
        # although not the most efficient way to compute it.
        f1Mp1 = pin.SE3(np.eye(3), f1p1)
        jacobian1 = f1Mp1.actionInverse @ jacobian1

        ## Transport the jacobian of frame 2 into the jacobian associated to x2
        # Vector from frame 2 center to p2
        f2p2 = x2 - rdata.oMf[shape2.parentFrame].translation
        # The following 2 lines are the easiest way to understand the transformation
        # although not the most efficient way to compute it.
        f2Mp2 = pin.SE3(np.eye(3), f2p2)
        jacobian2 = f2Mp2.actionInverse @ jacobian2

        CP1_SE3 = pin.SE3.Identity()
        CP1_SE3.translation = x1

        CP2_SE3 = pin.SE3.Identity()
        CP2_SE3.translation = x2
        J = (x1 - x2).T / distance @ (jacobian1[:3] - jacobian2[:3])
        return J
    


def d_ddot_dq(rmodel, cmodel, x: np.ndarray, shape1_id, shape2_id):
        
        
        q = x[: rmodel.nq]
        vq = x[rmodel.nq :]
        aq = np.zeros(rmodel.nq)

        # Creating the data models
        rdata = rmodel.createData()
        cdata = cmodel.createData()

        # Updating the position of the joints & the geometry objects.
        pin.forwardKinematics(rmodel, rdata, q, vq)
        pin.computeForwardKinematicsDerivatives(
            rmodel, rdata, q, vq, np.zeros(rmodel.nq)
        )
        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)

        # Poses and geometries of the shapes
        shape1_placement = cdata.oMg[shape1_id]
        shape2_placement = cdata.oMg[shape2_id]

        shape1 = cmodel.geometryObjects[shape1_id]
        shape2 = cmodel.geometryObjects[shape2_id]

        req = hppfcl.DistanceRequest()
        req.gjk_max_iterations = 20000
        req.abs_err = 0
        req.gjk_tolerance = 1e-9
        res = hppfcl.DistanceResult()
        
        
        distance = hppfcl.distance(
            shape1.geometry,
            shape1_placement,
            shape2.geometry,
            shape2_placement,
            req,
            res,
        )
        x1 = res.getNearestPoint1()
        x2 = res.getNearestPoint2()

        c1 = shape1_placement.translation
        c2 = shape2_placement.translation

        # Get body velocity at step 0
        pin.forwardKinematics(rmodel,rdata,q,vq)
        nu1 = pin.getFrameVelocity(rmodel,rdata,shape1.parentFrame,pin.LOCAL_WORLD_ALIGNED).copy()
        v1,w1 = nu1.linear,nu1.angular
        nu2 = pin.getFrameVelocity(rmodel,rdata,shape2.parentFrame,pin.LOCAL_WORLD_ALIGNED).copy()
        v2,w2 = nu2.linear,nu2.angular

        D1 = np.diagflat(shape1.geometry.radii)
        D2 = np.diagflat(shape2.geometry.radii)

        R1 = shape1_placement.rotation
        R2 = shape2_placement.rotation
        A1, A2 = R1 @ D1 @ R1.T, R2 @ D2 @ R2.T  # From pinocchio A = RDR.T

        sol_lam1, sol_lam2 = -(x1 - c1).T @ (x1 - x2), (x2 - c2).T @ (x1 - x2)

        Ldot = (
            (x1 - x2) @ (v1 - v2)
            - np.cross(x1 - x2, x1 - c1) @ w1
            + np.cross(x1 - x2, x2 - c2) @ w2
        )

        # Ldot = (x1 - x2).T @ (v1 - v2 - np.cross((x1 - c1), w1) + np.cross((x2 - c2), w2))
        dist_dot = Ldot / distance

        Lyy = np.r_[
            np.c_[np.eye(3) + sol_lam1 * A1, -np.eye(3), A1 @ (x1 - c1), np.zeros(3)],
            np.c_[-np.eye(3), np.eye(3) + sol_lam2 * A2, np.zeros(3), A2 @ (x2 - c2)],
            [np.r_[A1 @ (x1 - c1), np.zeros(3), np.zeros(2)]],
            [np.r_[np.zeros(3), A2 @ (x2 - c2), np.zeros(2)]],
        ]  ###! OK
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

        dx1 = np.c_[yc[:3], yr[:3]]
        dx2 = np.c_[yc[3:6], yr[3:6]]
        
        
                # theta = (c1,c2,r1,r2)
        theta_dot = np.r_[v1,w1,v2,w2]
        # dist_dot derivative wrt theta
        dL_dtheta = np.r_[x1-x2,-np.cross(x1-x2,x1-c1),-(x1-x2),np.cross(x1-x2,x2-c2)]
        ddL_dtheta2 = np.r_[dx1-dx2,
                        -pin.skew(x1-x2)@dx1+pin.skew(x1-c1)@(dx1-dx2),
                        -dx1+dx2,
                        pin.skew(x1-x2)@dx2-pin.skew(x2-c2)@(dx1-dx2)] \
                        +np.r_[np.zeros([3,12]),
                            np.c_[pin.skew(x1-x2),np.zeros([3,9])],
                            np.zeros([3,12]),
                            np.c_[np.zeros([3,6]),-pin.skew(x1-x2),np.zeros([3,3])]]
        # Verif using finite diff

        d_dist_dot_dtheta = (theta_dot.T@ddL_dtheta2/distance - dist_dot/distance**2 * dL_dtheta)

        # ### Derivative with respect to velocity: d_dist_dot_dthetadot
        d_dist_dot_dtheta_dot = dL_dtheta / distance

        # ##################################3
        # Robot derivatives

        pin.computeJointJacobians(rmodel,rdata,q)
        J1 = pin.getFrameJacobian(rmodel,rdata,shape1.parentFrame,pin.LOCAL_WORLD_ALIGNED)
        J2 = pin.getFrameJacobian(rmodel,rdata,shape2.parentFrame,pin.LOCAL_WORLD_ALIGNED)
        assert(np.allclose(nu1.vector,J1@vq))
        assert(np.allclose(nu2.vector,J2@vq))

        dtheta_dq = np.r_[J1,J2]
        assert(np.allclose(dtheta_dq@vq,theta_dot))

        dtheta_dot_dqdot = np.r_[J1,J2]

        pin.computeForwardKinematicsDerivatives(rmodel,rdata,q,vq,aq)

        in1_dnu1_dq,in1_dnu1_dqdot = pin.getFrameVelocityDerivatives(rmodel,rdata,shape1.parentFrame,pin.LOCAL)
        in2_dnu2_dq,in2_dnu2_dqdot = pin.getFrameVelocityDerivatives(rmodel,rdata,shape2.parentFrame,pin.LOCAL)

        inLWA1_dv1_dq =  R1@in1_dnu1_dq[:3] - pin.skew(v1)@R1@in1_dnu1_dqdot[3:]
        inLWA1_dw1_dq =  R1@in1_dnu1_dq[3:]
        inLWA2_dv2_dq =  R2@in2_dnu2_dq[:3] - pin.skew(v2)@R2@in2_dnu2_dqdot[3:]
        inLWA2_dw2_dq =  R2@in2_dnu2_dq[3:]
        
        dtheta_dot_dq = np.r_[inLWA1_dv1_dq,inLWA1_dw1_dq,inLWA2_dv2_dq,inLWA2_dw2_dq]

        # TODO: here a 0* is needed. WHHHHYYYYYYYYYYYYYYYYYYYYY!
        
        d_dist_dot_dq = d_dist_dot_dtheta @ dtheta_dq + d_dist_dot_dtheta_dot @ dtheta_dot_dq
        d_dist_dot_dqdot = d_dist_dot_dtheta_dot @ dtheta_dot_dqdot
        return d_dist_dot_dq,d_dist_dot_dqdot


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
