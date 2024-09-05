import unittest
import numpy as np

import hppfcl
import pinocchio as pin

from derivatives_X_computation import DerivativeComputation


class TestDistOpt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Define initial positions for the centers of the two ellipsoids
        cls.c1 = np.random.randn(3)
        cls.c2 = 10 * np.random.randn(3) + 10
        cls.center = np.concatenate((cls.c1, cls.c2))

        # Define the radii for the ellipsoids
        cls.radiiA = [2, 1, 1]
        cls.radiiB = [1, 2, 1]

        cls.shape1 = hppfcl.Ellipsoid(*cls.radiiA)
        cls.shape2 = hppfcl.Ellipsoid(*cls.radiiB)

        # Construct matrices A and B for ellipsoid constraints
        cls.D1 = np.diag([1 / r**2 for r in cls.radiiA])
        cls.D2 = np.diag([1 / r**2 for r in cls.radiiB])

        # Generate random rotation matrices using Pinocchio
        cls.R_1 = pin.SE3.Random().rotation
        cls.R_2 = pin.SE3.Random().rotation

        # Calculate rotated matrices
        cls.A1 = cls.R_1 @ cls.D1 @ cls.R_1.T
        cls.A2 = cls.R_2 @ cls.D2 @ cls.R_2.T

        cls.x = np.random.random(6)
        cls.lambda_ = np.random.random(2)

        cls.c1_SE3 = pin.SE3(rotation=cls.R_1, translation=cls.c1)
        cls.c2_SE3 = pin.SE3(rotation=cls.R_2, translation=cls.c2)
        

        cls.derivativeComputation = DerivativeComputation()

        cls.Lx_ND = numdiff(
            lambda variable: cls.derivativeComputation.L(
                variable, cls.center, cls.A1, cls.A2, cls.lambda_[0], cls.lambda_[1]
            ),
            cls.x,
        )
        cls.Lxx_ND = numdiff_matrix(
            lambda variable: cls.derivativeComputation.Lx(
                variable, cls.center, cls.A1, cls.A2, cls.lambda_[0], cls.lambda_[1]
            ),
            cls.x,
        )
        cls.M_ND = numdiff_matrix(
            lambda variable: cls.derivativeComputation.K(
                variable, cls.center, cls.A1, cls.A2
            ),
            np.concatenate((cls.x, cls.lambda_)),
        )
        cls.dh1_dx_ND = numdiff(
            lambda variable: cls.derivativeComputation.h1(variable, cls.center, cls.A1),
            cls.x,
        )
        cls.dh2_dx_ND = numdiff(
            lambda variable: cls.derivativeComputation.h2(variable, cls.center, cls.A2),
            cls.x,
        )
        cls.dh1_dcenter_ND = numdiff(
            lambda variable: cls.derivativeComputation.h1(cls.x, variable, cls.A1),
            cls.center,
        )
        cls.dh2_dcenter_ND = numdiff(
            lambda variable: cls.derivativeComputation.h2(cls.x, variable, cls.A2),
            cls.center,
        )
        cls.dh1_R_ND = numdiff_rot(
            lambda variable: cls.derivativeComputation.h1(cls.x, cls.center, variable),
            cls.A1,
        )
        cls.dh2_R_ND = numdiff_rot(
            lambda variable: cls.derivativeComputation.h2(cls.x, cls.center, variable),
            cls.A2,
        )

    def test_Lx(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.Lx_ND
                - cls.derivativeComputation.Lx(
                    cls.x, cls.center, cls.A1, cls.A2, cls.lambda_[0], cls.lambda_[1]
                )
            ),
            0,
            places=4,
            msg="The value of the derivative of the Lagrangian w.r.t. x is not equal to the finite different one.",
        )

    def test_Lxx(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.Lxx_ND - cls.derivativeComputation.Lxx(cls.lambda_, cls.A1, cls.A2)
            ),
            0,
            places=4,
            msg="The value of the Hessian of the Lagrangian w.r.t. x is not equal to the finite different one.",
        )

    def test_M(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.M_ND
                - cls.derivativeComputation.M(
                    np.concatenate((cls.x, cls.lambda_)), cls.center, cls.A1, cls.A2
                )
            ),
            0,
            places=4,
            msg="The value of the derivative of the KKT matrix w.r.t. x is not equal to the finite different one.",
        )

    def test_dh1_dx(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh1_dx_ND
                - cls.derivativeComputation.dh1_dx(cls.x, cls.center, cls.A1)
            ),
            0,
            places=4,
            msg="The value of the derivative of the hard constraint h1 w.r.t. the closest point 1 is not equal to the finite different one.",
        )

    def test_dh2_dx(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh2_dx_ND
                - cls.derivativeComputation.dh2_dx(cls.x, cls.center, cls.A2)
            ),
            0,
            places=4,
            msg="The value of the derivative of the hard constraint h2 w.r.t. the closest point 1 is not equal to the finite different one.",
        )

    def test_dh1_dcenter(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh1_dcenter_ND
                - cls.derivativeComputation.dh1_dcenter(cls.x, cls.center, cls.A1)
            ),
            0,
            places=4,
            msg="The value of the derivative of the hard constraint h1 w.r.t. the center of the ellipsoids is not equal to the finite different one.",
        )

    def test_dh2_dcenter(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh2_dcenter_ND
                - cls.derivativeComputation.dh2_dcenter(cls.x, cls.center, cls.A2)
            ),
            0,
            places=4,
            msg="The value of the derivative of the hard constraint h2 w.r.t. the center of the ellipsoids is not equal to the finite different one.",
        )

    def test_dh1_dR(cls):
        
        w = np.random.random(3)
        dt = 1e-6
        ht = cls.derivativeComputation.h1(cls.x, cls.center, cls.A1)
        ht_plus = cls.derivativeComputation.h1(cls.x, cls.center, pin.exp3(w * dt) @ cls.R_1 @  cls.D1 @ (pin.exp3(w * dt) @ cls.R_1).T)
        dhdt = cls.derivativeComputation.dh1_dR(cls.x, cls.center, cls.A1)[:3].T @ w
        
        cls.assertAlmostEqual(
            np.linalg.norm(
                (ht - ht_plus) / dt - dhdt
            ),
            0,
            places=5,
            msg="The value of the derivative of the hard constraint h1 w.r.t. the rotations of the ellipsoids is not equal to the finite different one.",
        )

    def test_dh2_dR(cls):
        
        w = np.random.random(3)
        dt = 1e-6
        ht = cls.derivativeComputation.h2(cls.x, cls.center, cls.A2)
        ht_plus = cls.derivativeComputation.h2(cls.x, cls.center, cls.R_2 @ pin.exp3(w * dt) @ cls.D2 @ cls.R_2.T)
        dhdt = cls.derivativeComputation.dh2_dR(cls.x, cls.center, cls.A2)[:3].T @ w
        
        cls.assertAlmostEqual(
            np.linalg.norm(
                (ht - ht_plus) / dt - dhdt
            ),
            0,
            places=5,
            msg="The value of the derivative of the hard constraint h1 w.r.t. the rotations of the ellipsoids is not equal to the finite different one.",
        )
    


def numdiff(f, inX, h=1e-8):
    # Computes the Jacobian of a function returning a 1d array
    f0 = np.array(f(inX)).copy()
    x = inX.copy()
    df_dx = np.zeros(len(x))
    for ix in range(len(x)):
        x[ix] += h
        df_dx[ix] = (f(x) - f0) / h
        x[ix] = inX[ix]
    return df_dx


def numdiff_matrix(f, inX, h=1e-6):
    f0 = np.array(f(inX)).copy()
    x = inX.copy()
    df_dx = np.zeros((f0.size, len(x)))
    for ix in range(len(x)):
        x[ix] += h
        df_dx[:, ix] = (f(x) - f0) / h
        x[ix] = inX[ix]
    return df_dx


def numdiff_rot(f, inR, h=1e-8):
    f0 = np.array(f(inR)).copy()
    R = inR.copy()
    df_dR = np.zeros(3)
    for iw in range(3):
        eps = np.zeros(3)
        eps[iw] = h
        R =pin.exp3(eps) @ R
        df_dR[iw] = (f(R) - f0) / h
        R = inR
    return df_dR


if __name__ == "__main__":
    unittest.main()
