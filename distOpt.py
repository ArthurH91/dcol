from typing import Tuple
import unittest
import numpy as np
from qcqp_solver import EllipsoidOptimization


class DistOpt:

    def __init__(self) -> None:
        pass

    def set_up_ellips(
        self,
        A: np.ndarray,
        B: np.ndarray,
        x01: np.ndarray,
        x02: np.ndarray,
        J01: np.ndarray,
        J02: np.ndarray,
    ):
        """
        Sets up the ellipsoids with their curvature matrices and centers.

        Args:
            A (np.array): (3, 3) array describing the curvature of the first ellipsoid.
            B (np.array): (3, 3) array describing the curvature of the second ellipsoid.
            x01 (np.array): (3, 1) array, center of the first ellipsoid.
            x02 (np.array): (3, 1) array, center of the second ellipsoid.
            J01 (np.array): (3, nq) array, Jacobian of the first ellipsoid with regards to the configuration of the robot.
            J02 (np.array): (3, nq) array, Jacobian of the second ellipsoid with regards to the configuration of the robot.

        Raises:
            ValueError: If the input matrices or vectors do not have the correct dimensions.
        """
        if A.shape != (3, 3) or B.shape != (3, 3):
            raise ValueError("A and B must be 3x3 matrices.")
        if x01.shape != (3,) or x02.shape != (3,):
            raise ValueError("x01 and x02 must be 3-element vectors.")
        if J01.shape[0] != 3 or J02.shape[0] != 3:
            raise ValueError("J01 and J02 must have 3 rows.")

        self.A = A
        self.B = B
        self.x01 = x01
        self.x02 = x02
        self.J01 = J01
        self.J02 = J02

    def set_up_optim_var(
        self, x1: np.ndarray, x2: np.ndarray, d: float, lambda1: float, lambda2: float
    ):
        """
        Sets up the optimization variables.

        Args:
            x1 (np.array): (3, 1) array, solution variable for the first ellipsoid.
            x2 (np.array): (3, 1) array, solution variable for the second ellipsoid.
            d (float): Scalar, solution of the QCQP.
            lambda1 (float): Lagrange multiplier associated with the first ellipsoid.
            lambda2 (float): Lagrange multiplier associated with the second ellipsoid.

        Raises:
            ValueError: If the input vectors do not have the correct dimensions.
        """
        if x1.shape != (3,) or x2.shape != (3,):
            raise ValueError("x1 and x2 must be 3-element vectors.")
        if (
            not isinstance(d, (int, float))
            or not isinstance(lambda1, (int, float))
            or not isinstance(lambda2, (int, float))
        ):
            raise ValueError("d, lambda1, and lambda2 must be scalars.")

        self.x1 = x1
        self.x2 = x2
        self.d = d
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def gradient_xL(self) -> np.ndarray:
        """
        Computes the gradient of the Lagrangian with respect to x = (x1, x2).T

        Returns:
            np.array: Gradient of the Lagrangian.
        """
        Dxl = np.zeros((6,))

        Dxl[:3] = (1 / self.d) * (self.x1 - self.x2) + self.lambda1 * np.matmul(
            self.A, (self.x1 - self.x01)
        )
        Dxl[3:] = -(1 / self.d) * (self.x1 - self.x2) + self.lambda2 * np.matmul(
            self.B, (self.x2 - self.x02).T
        )
        return Dxl

    def hessian_xxL(self) -> np.ndarray:
        """
        Computes the Hessian of the Lagrangian with respect to x = (x1, x2).T

        Returns:
            np.array: Hessian of the Lagrangian w.rt. x = (x1, x2).T.
        """
        DDxl = np.zeros((6, 6))

        DDxl[:3, :3] = 1 / self.d * np.eye(3, 3) + self.lambda1 * self.A
        DDxl[:3, 3:] = -1 / self.d * np.eye(3, 3)
        DDxl[-3:, :3] = -1 / self.d * np.eye(3, 3)
        DDxl[3:, 3:] = 1 / self.d * np.eye(3, 3) + self.lambda2 * self.B

        return DDxl

    def gradient_xh1(self) -> np.ndarray:
        """
        Computes the gradient of the first constraint with regards to the position of the closest points.

        Returns:
            np.array: Gradient of the first constraint.
        """
        DxH1 = np.zeros((6,))
        DxH1[:3] = np.dot(self.A, (self.x1 - self.x01))
        return DxH1

    def gradient_xh2(self) -> np.ndarray:
        """
        Computes the gradient of the second constraint with regards to the position of the closest points.

        Returns:
            np.array: Gradient of the second constraint.
        """
        DxH2 = np.zeros((6,))
        DxH2[3:] = np.dot(self.B, (self.x2 - self.x02))
        return DxH2

    def gradient_qh1(self) -> np.ndarray:
        """
        Computes the gradient of the first constraint with regards to the configuration of the robot.

        Returns:
            np.array: Gradient of the first constraint.
        """
        DqH1 = np.dot(np.dot(self.J01.T, self.A), (self.x1 - self.x01))
        return DqH1

    def gradient_qh2(self) -> np.ndarray:
        """
        Computes the gradient of the second constraint with regards to the configuration of the robot.

        Returns:
            np.array: Gradient of the second constraint.
        """
        DqH2 = np.dot(np.dot(self.J02.T, self.B), (self.x2 - self.x02))
        return DqH2

    def hessian_qx_L(self) -> np.ndarray:
        """
        Computes the Hessian of the Lagrangian with respect to x = (x1, x2).T and q.

        Returns:
            np.array: Hessian of the Lagrangian w.rt. x = (x1, x2).T and q.
        """
        DDqxl = np.zeros((6, 7))

        DDqxl[:3, :] = self.lambda1 * np.dot(self.A, self.J01)
        DDqxl[3:, :] = self.lambda2 * np.dot(self.B, self.J02)

        return DDqxl

    def M(self) -> np.ndarray:
        """
        Constructs the M0 matrix used in the optimization process.

        Returns:
            np.array: M0 matrix.
        """
        M = np.zeros((8, 8))
        M[:6, :6] = self.hessian_xxL()
        M[:6, 6] = self.gradient_xh1()
        M[:6, 7] = self.gradient_xh2()
        M[6, :6] = self.gradient_xh1().T
        M[7, :6] = self.gradient_xh2().T
        return M

    def N(self) -> np.ndarray:
        """
        Constructs the N matrix used in the optimization process.

        Returns:
            np.array: N matrix.
        """
        N = np.zeros((8, 7))

        N[:6, :] = self.hessian_qx_L()
        N[6, :] = self.gradient_qh1()
        N[7, :] = self.gradient_qh2()

        return N

    def invMN(self) -> np.ndarray:
        """
        Computes the product of the inverse of M matrix and N matrix.

        Returns:
            np.ndarray: Result of inv(M) * N.
        """
        try:
            return np.dot(np.linalg.inv(self.M()), self.N())
        except np.linalg.LinAlgError as e:
            raise ValueError("Matrix M is singular and cannot be inverted.") from e

    def get_dY_dq(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        d: float,
        lambda1: float,
        lambda2: float,
        A: np.ndarray,
        B: np.ndarray,
        x01: np.ndarray,
        x02: np.ndarray,
        J01: np.ndarray,
        J02: np.ndarray,
    ) -> Tuple[np.ndarray]:
        """
        Computes the derivatives of the optimization variables with respect to the configuration variables.

        Args:
            x1 (np.array): (3, 1) array, solution variable for the first ellipsoid.
            x2 (np.array): (3, 1) array, solution variable for the second ellipsoid.
            d (float): Scalar, solution of the QCQP.
            lambda1 (float): Lagrange multiplier associated with the first ellipsoid.
            lambda2 (float): Lagrange multiplier associated with the second ellipsoid.
            A (np.array): (3, 3) array describing the curvature of the first ellipsoid.
            B (np.array): (3, 3) array describing the curvature of the second ellipsoid.
            x01 (np.array): (3, 1) array, center of the first ellipsoid.
            x02 (np.array): (3, 1) array, center of the second ellipsoid.
            J01 (np.array): (3, nq) array, Jacobian of the first ellipsoid with regards to the configuration of the robot.
            J02 (np.array): (3, nq) array, Jacobian of the second ellipsoid with regards to the configuration of the robot.

        Returns:
            Tuple[np.ndarray]: Derivatives of the optimization variables with respect to the configuration variables.
        """
        self.set_up_ellips(A, B, x01, x02, J01, J02)
        self.set_up_optim_var(x1, x2, d, lambda1, lambda2)
        dY_dq = self.invMN()
        dx1_dq = dY_dq[:3, :]
        dx2_dq = dY_dq[3:6, :]
        return dx1_dq, dx2_dq


class TestDistOpt(unittest.TestCase):

    def setUp(self):
        self.A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        self.B = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
        self.x01 = np.array([1, 1, 1])
        self.x02 = np.array([5, 5, 5])

        # Initialize the QCQPSolver with the ellipsoid parameters
        self.qcqp_solver = EllipsoidOptimization()
        self.qcqp_solver.setup_problem(self.x01, self.A, self.x02, self.B)
        self.qcqp_solver.solve_problem(
            warm_start_primal=np.concatenate((self.x01, self.x02))
        )

        self.x1, self.x2 = self.qcqp_solver.get_optimal_values()
        self.distance = self.qcqp_solver.get_minimum_cost()

        self.lambda1, self.lambda2 = self.qcqp_solver.get_dual_values()

        self.J1 = np.random.rand(3, 7)
        self.J2 = np.random.rand(3, 7)

        self.opt = DistOpt()
        self.opt.set_up_ellips(self.A, self.B, self.x01, self.x02, self.J1, self.J2)
        self.opt.set_up_optim_var(
            self.x1, self.x2, self.distance, self.lambda1, self.lambda2
        )

    def test_M(self):

        M = self.opt.M()
        print(M.shape)

    def test_N(self):
        N = self.opt.N()
        print(N.shape)

    def test_MN(self):
        MN = self.opt.invMN()
        print(MN.shape)


if __name__ == "__main__":
    # unittest.main()

    test = TestDistOpt()
    test.setUp()
    test.test_M()
    test.test_N()
    test.test_MN()
