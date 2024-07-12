import unittest

import numpy as np

class DistOpt:
    
    def __init__(self) -> None:
        pass

    def set_up_ellips(self, A, B, x01, x02):
        """
        Sets up the ellipsoids with their curvature matrices and centers.

        Args:
            A (np.array): (3, 3) array describing the curvature of the first ellipsoid.
            B (np.array): (3, 3) array describing the curvature of the second ellipsoid.
            x01 (np.array): (3, 1) array, center of the first ellipsoid.
            x02 (np.array): (3, 1) array, center of the second ellipsoid.
        """
        self.A = A
        self.B = B
        self.x01 = x01
        self.x02 = x02

    def set_up_optim_var(self, x1, x2, d, lambda1, lambda2):
        """
        Sets up the optimization variables.

        Args:
            x1 (np.array): (3, 1) array, solution variable for the first ellipsoid.
            x2 (np.array): (3, 1) array, solution variable for the second ellipsoid.
            d (float): Scalar, solution of the QCQP.
            lambda1 (float): Lagrange multiplier associated with the first ellipsoid.
            lambda2 (float): Lagrange multiplier associated with the second ellipsoid.
        """
        self.x1 = x1
        self.x2 = x2
        self.d = d
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def gradient_xL(self):
        """
        Computes the gradient of the Lagrangian with respect to x = (x1, x2).T

        Returns:
            np.array: Gradient of the Lagrangian.
        """
        Dxl = np.zeros((6, 1))
        Dxl[:3] = (1 / self.d) * (self.x1 - self.x2) + self.lambda1 * np.dot(self.A, (self.x1 - self.x01).T)
        Dxl[3:] = -(1 / self.d) * (self.x1 - self.x2) + self.lambda2 * np.dot(self.B, (self.x2 - self.x02).T)
        return Dxl

    def gradient_xh1(self):
        """
        Computes the gradient of the first constraint.

        Returns:
            np.array: Gradient of the first constraint.
        """
        DxH1 = np.zeros((6, 1))
        DxH1[:3] = np.dot(self.A, (self.x1 - self.x01))
        return DxH1

    def gradient_xh2(self):
        """
        Computes the gradient of the second constraint.

        Returns:
            np.array: Gradient of the second constraint.
        """
        DxH2 = np.zeros((6, 1))
        DxH2[3:] = np.dot(self.B, (self.x2 - self.x02))
        return DxH2

    def M0(self):
        """
        Constructs the M0 matrix used in the optimization process.

        Returns:
            np.array: M0 matrix.
        """
        M = np.zeros((18, 3))
        M[:6, 0] = self.gradient_xL()
        M[:6, 1] = self.gradient_xh1()
        M[:6, 2] = self.gradient_xh2()
        M[6:12, 0] = self.gradient_xh1()
        M[12:18, 0] = self.gradient_xh2()
        return M


class TestDistOpt(unittest.TestCase):
    
    def setUp(self):
        self.A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        self.B = np.array([[4, 0, 0], [0, 5, 0], [0, 0, 6]])
        self.x01 = np.array([[1], [1], [1]])
        self.x02 = np.array([[2], [2], [2]])

        self.x1 = np.array([[1.5], [1.5], [1.5]])
        self.x2 = np.array([[2.5], [2.5], [2.5]])
        self.d = 1.0
        self.lambda1 = 0.5
        self.lambda2 = 0.5

        self.opt = DistOpt()
        self.opt.set_up_ellips(self.A, self.B, self.x01, self.x02)
        self.opt.set_up_optim_var(self.x1, self.x2, self.d, self.lambda1, self.lambda2)

    def test_set_up_ellips(self):
        self.assertTrue(np.array_equal(self.opt.A, self.A))
        self.assertTrue(np.array_equal(self.opt.B, self.B))
        self.assertTrue(np.array_equal(self.opt.x01, self.x01))
        self.assertTrue(np.array_equal(self.opt.x02, self.x02))

    def test_set_up_optim_var(self):
        self.assertTrue(np.array_equal(self.opt.x1, self.x1))
        self.assertTrue(np.array_equal(self.opt.x2, self.x2))
        self.assertEqual(self.opt.d, self.d)
        self.assertEqual(self.opt.lambda1, self.lambda1)
        self.assertEqual(self.opt.lambda2, self.lambda2)

    def test_gradient_xL(self):
        expected_gradient_xL = np.zeros((6, 1))
        expected_gradient_xL[:3] = (1 / self.d) * (self.x1 - self.x2) + self.lambda1 * np.dot(self.A, (self.x1 - self.x01).T)
        expected_gradient_xL[3:] = -(1 / self.d) * (self.x1 - self.x2) + self.lambda2 * np.dot(self.B, (self.x2 - self.x02).T)
        
        gradient_xL = self.opt.gradient_xL()
        np.testing.assert_array_almost_equal(gradient_xL, expected_gradient_xL)

    def test_gradient_xh1(self):
        expected_gradient_xh1 = np.zeros((6, 1))
        expected_gradient_xh1[:3] = np.dot(self.A, (self.x1 - self.x01))
        
        gradient_xh1 = self.opt.gradient_xh1()
        np.testing.assert_array_almost_equal(gradient_xh1, expected_gradient_xh1)

    def test_gradient_xh2(self):
        expected_gradient_xh2 = np.zeros((6, 1))
        expected_gradient_xh2[3:] = np.dot(self.B, (self.x2 - self.x02))
        
        gradient_xh2 = self.opt.gradient_xh2()
        np.testing.assert_array_almost_equal(gradient_xh2, expected_gradient_xh2)

    def test_M0(self):
        M = np.zeros((18, 3))
        M[:6, 0] = self.opt.gradient_xL().flatten()
        M[:6, 1] = self.opt.gradient_xh1().flatten()
        M[:6, 2] = self.opt.gradient_xh2().flatten()
        M[6:12, 0] = self.opt.gradient_xh1().flatten()
        M[12:18, 0] = self.opt.gradient_xh2().flatten()
        
        M0 = self.opt.M0()
        np.testing.assert_array_almost_equal(M0, M)

if __name__ == '__main__':
    unittest.main()
