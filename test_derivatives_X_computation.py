import unittest
import numpy as np

import hppfcl
import pinocchio as pin

from ellipsoid_optimization import EllipsoidOptimization
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
        D1 = np.diag([1 / r**2 for r in cls.radiiA])
        D2 = np.diag([1 / r**2 for r in cls.radiiB])

        # Generate random rotation matrices using Pinocchio
        cls.R_A = pin.SE3.Random().rotation
        cls.R_B = pin.SE3.Random().rotation

        # Calculate rotated matrices
        cls.A1 = cls.R_A.T @ D1 @ cls.R_A
        cls.A2 = cls.R_B.T @ D2 @ cls.R_B

        cls.x = np.random.random(6)
        cls.lambda_ = np.random.random(2)

        cls.c1_SE3 = pin.SE3(rotation=cls.R_A.T, translation=cls.c1)
        cls.c2_SE3 = pin.SE3(rotation=cls.R_B.T, translation=cls.c2)

        cls.derivativeComputation = DerivativeComputation()


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

    def test_dh1_dx1(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh1_dx_ND[0,:3]
                - cls.derivativeComputation.dh1_dx(cls.x, cls.center, cls.A1)[:3,0]
            ),
            0,
            places=5,
            msg="The value of the derivative of the hard constraint h1 w.r.t. the closest point 1 is not equal to the finite different one.",
        )
        
    def test_dh1_dx2(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh1_dx_ND[0,3:]
                - cls.derivativeComputation.dh1_dx(cls.x, cls.center, cls.A1)[:3,1]
            ),
            0,
            places=5,
            msg="The value of the derivative of the hard constraint h1 w.r.t. the closest point 2 is not equal to the finite different one.",
        )

    def test_dh2_dx1(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh2_dx_ND[0,:3]
                - cls.derivativeComputation.dh2_dx(cls.x, cls.center, cls.A2)[:3,0]
            ),
            0,
            places=5,
            msg="The value of the derivative of the hard constraint h2 w.r.t. the closest point 1 is not equal to the finite different one.",
        )


    def test_dh2_dx2(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh2_dx_ND[0,3:]
                - cls.derivativeComputation.dh2_dx(cls.x, cls.center, cls.A2)[:3,1]
            ),
            0,
            places=5,
            msg="The value of the derivative of the hard constraint h2 w.r.t. the closest point 2 is not equal to the finite different one.",
        )

    def test_dh1_dcenter1(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh1_dcenter_ND[0,:3]
                - cls.derivativeComputation.dh1_dcenter(cls.x, cls.center, cls.A1)[:3,0]
            ),
            0,
            places=5,
            msg="The value of the derivative of the hard constraint h1 w.r.t. the center of the ellipsoids is not equal to the finite different one.",
        )

    def test_dh1_dcenter2(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh1_dcenter_ND[0,3:]
                - cls.derivativeComputation.dh1_dcenter(cls.x, cls.center, cls.A1)[:3,1]
            ),
            0,
            places=5,
            msg="The value of the derivative of the hard constraint h1 w.r.t. the center of the ellipsoids is not equal to the finite different one.",
        )

    def test_dh2_dcenter1(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh2_dcenter_ND[0,:3]
                - cls.derivativeComputation.dh2_dcenter(cls.x, cls.center, cls.A1)[:3,0]
            ),
            0,
            places=5,
            msg="The value of the derivative of the hard constraint h2 w.r.t. the center of the ellipsoid 1 is not equal to the finite different one.",
        )

    def test_dh2_dcenter2(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh1_dcenter_ND[0,3:]
                - cls.derivativeComputation.dh1_dcenter(cls.x, cls.center, cls.A1)[:3,1]
            ),
            0,
            places=5,
            msg="The value of the derivative of the hard constraint h2 w.r.t. the center of the ellipsoid 2is not equal to the finite different one.",
        )
        
    # def test_dh1_dR(cls):

    #     cls.assertAlmostEqual(
    #         np.linalg.norm(
    #             cls.dh1_dcenter_ND
    #             - cls.derivativeComputation.dh1_dcenter(cls.x, cls.center, cls.A1)
    #         ),
    #         0,
    #         places=5,
    #         msg="The value of the derivative of the hard constraint h1 w.r.t. the center of the ellipsoids is not equal to the finite different one.",
    #     )

    # def test_dh2_dR(cls):

    #     cls.assertAlmostEqual(
    #         np.linalg.norm(
    #             cls.dh1_dcenter_ND
    #             - cls.derivativeComputation.dh1_dcenter(cls.x, cls.center, cls.A1)
    #         ),
    #         0,
    #         places=5,
    #         msg="The value of the derivative of the hard constraint h2 w.r.t. the center of the ellipsoids is not equal to the finite different one.",
    #     )
        


def numdiff(f, inX, h=1e-6):
    # Computes the Jacobian of a function returning a 1d array
    f0 = np.array(f(inX)).copy()
    x = inX.copy()
    df_dx = np.zeros((f0.size, len(x)))
    for ix in range(len(x)):
        x[ix] += h
        df_dx[:, ix] = (f(x) - f0) / h
        x[ix] = inX[ix]
    return df_dx


if __name__ == "__main__":
    unittest.main()
