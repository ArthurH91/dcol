from typing import Tuple
import unittest
import numpy as np
from qcqp_solver import EllipsoidOptimization

np.set_printoptions(3)


class DistOpt:

    # def __init__(self) -> None:
    #     pass

    def __init__(self) -> None:
        self.Dxl = np.zeros((6,))
        self.DDxl = np.zeros((6, 6))
        self.DxH1 = np.zeros((6,))
        self.DxH2 = np.zeros((6,))
        self.DoH1 = np.zeros((6,))
        self.DoH2 = np.zeros((6,))
        self.DDoxl = np.zeros((6, 6))
        self.M_matrix = np.zeros((8, 8))
        self.N_matrix = np.zeros((8, 6))

    def set_up_ellips(
        self,
        A: np.ndarray,
        B: np.ndarray,
        x01: np.ndarray,
        x02: np.ndarray,
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

        self.A = A
        self.B = B
        self.x01 = x01
        self.x02 = x02

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
            not isinstance(d, (float))
            or not isinstance(lambda1, (float))
            or not isinstance(lambda2, (float))
        ):
            raise ValueError("d, lambda1, and lambda2 must be floats.")

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

        self.Dxl[:3] = (self.x1 - self.x2) + self.lambda1 * np.matmul(
            self.A, (self.x1 - self.x01)
        )
        self.Dxl[3:] = -(self.x1 - self.x2) + self.lambda2 * np.matmul(
            self.B, (self.x2 - self.x02).T
        )
        return self.Dxl

    def hessian_xxL(self) -> np.ndarray:
        """
        Computes the Hessian of the Lagrangian with respect to x = (x1, x2).T

        Returns:
            np.array: Hessian of the Lagrangian w.rt. x = (x1, x2).T.
        """
        self.DDxl[:3, :3] = np.eye(3, 3) + self.lambda1 * self.A
        self.DDxl[:3, 3:] = -np.eye(3, 3)
        self.DDxl[3:, :3] = -np.eye(3, 3)
        self.DDxl[3:, 3:] = np.eye(3, 3) + self.lambda2 * self.B

        return self.DDxl

    def gradient_xh1(self) -> np.ndarray:
        """
        Computes the gradient of the first constraint with regards to the position of the closest points.

        Returns:
            np.array: Gradient of the first constraint.
        """

        self.DxH1[:3] = self.lambda1 * np.matmul(self.A, (self.x1 - self.x01))
        return self.DxH1

    def gradient_xh2(self) -> np.ndarray:
        """
        Computes the gradient of the second constraint with regards to the position of the closest points.

        Returns:
            np.array: Gradient of the second constraint.
        """
        self.DxH2[3:] = self.lambda2 * np.matmul(self.B, (self.x2 - self.x02))
        return self.DxH2

    def gradient_oh1(self) -> np.ndarray:
        """
        Computes the gradient of the first constraint with regards to the configuration of the robot.

        Returns:
            np.array: Gradient of the first constraint.
        """
        self.DoH1[:3] = -self.lambda1 * np.matmul(self.A, (self.x1 - self.x01))
        return self.DoH1

    def gradient_oh2(self) -> np.ndarray:
        """
        Computes the gradient of the second constraint with regards to the configuration of the robot.

        Returns:
            np.array: Gradient of the second constraint.
        """
        self.DoH2[3:] = -self.lambda2 * np.matmul(self.B, (self.x2 - self.x02))
        return self.DoH2

    def hessian_ox_L(self) -> np.ndarray:
        """
        Computes the Hessian of the Lagrangian with respect to x = (x1, x2).T and q.

        Returns:
            np.array: Hessian of the Lagrangian w.rt. x = (x1, x2).T and q.
        """

        self.DDoxl[:3, :3] = -self.lambda1 * self.A
        self.DDoxl[3:, 3:] = -self.lambda2 * self.B

        return self.DDoxl

    def M(self) -> np.ndarray:
        """
        Constructs the M0 matrix used in the optimization process.

        Returns:
            np.array: M0 matrix.
        """
        self.M_matrix[:6, :6] = self.hessian_xxL()
        self.M_matrix[:6, 6] = self.gradient_xh1()
        self.M_matrix[:6, 7] = self.gradient_xh2()
        self.M_matrix[6, :6] = self.gradient_xh1().T
        self.M_matrix[7, :6] = self.gradient_xh2().T
        return self.M_matrix

    def N(self) -> np.ndarray:
        """
        Constructs the N matrix used in the optimization process.

        Returns:
            np.array: N matrix.
        """
        self.N_matrix[:6, :] = self.hessian_ox_L()
        self.N_matrix[6, :] = self.gradient_oh1()
        self.N_matrix[7, :] = self.gradient_oh2()
        return self.N_matrix

    def invMN(self) -> np.ndarray:
        """
        Computes the product of the inverse of M matrix and N matrix.

        Returns:
            np.ndarray: Result of inv(M) * N.
        """
        try:
            return -np.matmul(np.linalg.inv(self.M()), self.N())
        except np.linalg.LinAlgError as e:
            raise ValueError("Matrix M is singular and cannot be inverted.") from e

    def get_dY_dXo(
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

        Returns:
            Tuple[np.ndarray]: Derivatives of the optimization variables with respect to the configuration variables.
        """
        self.set_up_ellips(A, B, x01, x02)
        self.set_up_optim_var(x1, x2, d, lambda1, lambda2)
        self.set_matrices()
        dY_do = self.invMN()
        dx1_do = dY_do[:3, :]
        dx2_do = dY_do[3:6, :]
        return dx1_do, dx2_do


class TestDistOpt(unittest.TestCase):
    
    
    def setEllispoidsRadii(self):

        A = np.array([[1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
        B = np.array([[0.1, 0, 0], [0, 0.6, 0], [0, 0, 1]])

        return A,B
    
    def compute_gradient_xL_wrt_x0(self, x0):
        
        ### Setting up the problem
        
        # Setting up the variables of the ellipsoid minimal distance problem
        A, B = self.setEllispoidsRadii()
        
        # Centers of the ellipsoids
        x01 = x0[:3]
        x02 = x0[3:]
        
        print("------------- NUMDIFF ------------------ ")
        print(f"x01: {x01}")
        print(f"x02: {x02}")
        # Initialize the QCQPSolver with the ellipsoid parameters
        qcqp_solver = EllipsoidOptimization()
        qcqp_solver.setup_problem(x01, A, x02, B)
        qcqp_solver.solve_problem(warm_start_primal=x0)

        # Taking out the variables necessary for computing the gradient
        xSol1, xSol2 = qcqp_solver.get_optimal_values()
        
        print(f"xSol1: {xSol1}")
        print(f"xSol2: {xSol2}")
        distance = qcqp_solver.get_minimum_cost()
        lambda1, lambda2 = qcqp_solver.get_dual_values()
        
        
        print(f"lambda1 : {lambda1}")
        print(f"lambda2 : {lambda2}")
        ### Computing the gradient
        opt = DistOpt()
        opt.set_up_ellips(A, B, x01, x02)
        opt.set_up_optim_var(xSol1, xSol2, distance, lambda1, lambda2)
        
        print(f"xsol1 - xsol2 : {xSol1 - xSol2}")
        print(f"lambda1 * A: {lambda1 * A}")
        print(f"lambda1 * A @ (x1 -x01) : {lambda1 * np.matmul(A, (xSol1 - x01))}")
        print(f"numdiff opt.gradient_xL(): \n {opt.gradient_xL()} ")
        print("------------- END NUMDIFF ------------------ ")

        return opt.gradient_xL()
    
    def compute_Hessian_xx0L(self, x0):
        
        ### Setting up the problem
        
        # Setting up the variables of the ellipsoid minimal distance problem
        A, B = self.setEllispoidsRadii()
        
        # Centers of the ellipsoids
        x01 = x0[:3]
        x02 = x0[3:]
        
        # Initialize the QCQPSolver with the ellipsoid parameters
        qcqp_solver = EllipsoidOptimization()
        qcqp_solver.setup_problem(x01, A, x02, B)
        qcqp_solver.solve_problem(warm_start_primal=x0)

        # Taking out the variables necessary for computing the gradient
        xSol1, xSol2 = qcqp_solver.get_optimal_values()
        distance = qcqp_solver.get_minimum_cost()
        lambda1, lambda2 = qcqp_solver.get_dual_values()
        
        ### Computing the gradient
        opt = DistOpt()
        opt.set_up_ellips(A, B, x01, x02)
        opt.set_up_optim_var(xSol1, xSol2, distance, lambda1, lambda2)
        print(f"opt.gradient_xL(): \n {opt.gradient_xL()} ")

        return opt.hessian_ox_L()
    
    def test_numdiff_Hessian_xx0L(self):
        
        x0 = np.array([1.0,2.0,3.0,10.0,11.0,12.0])
        
        numdiff_hess = self.numdiff(self.compute_gradient_xL_wrt_x0, x0, 1e-6)
        print(f"numdiff_hess : \n{numdiff_hess}")
        hess = self.compute_Hessian_xx0L(x0)
        print(f"hess : \n{hess}")
        self.assertTrue(np.testing.assert_almost_equal(hess, numdiff_hess ))
        
        
    
    
        

    # def computeEllipsoidOpt(self, x0):

    #     x01 = x0[:3]
    #     x02 = x0[3:]
    #     # Initialize the QCQPSolver with the ellipsoid parameters
    #     qcqp_solver = EllipsoidOptimization()
    #     qcqp_solver.setup_problem(x01, self.A, x02, self.B)
    #     qcqp_solver.solve_problem(warm_start_primal=np.concatenate((x01, x02)))

    #     self.x1, self.x2 = qcqp_solver.get_optimal_values()
    #     self.distance = qcqp_solver.get_minimum_cost()

    #     self.lambda1, self.lambda2 = qcqp_solver.get_dual_values()

    # def getXSol(self, x0):

    #     self.computeEllipsoidOpt(x0)

    #     return np.concatenate((self.x1, self.x2))

    # def get_dY_dXo(self, x0):

    #     x01 = x0[:3]
    #     x02 = x0[3:]

    #     self.computeEllipsoidOpt(x0)
    #     self.opt = DistOpt()

    #     dx1_do, dx2_do = self.opt.get_dY_dXo(
    #         self.x1,
    #         self.x2,
    #         self.distance,
    #         self.lambda1,
    #         self.lambda2,
    #         self.A,
    #         self.B,
    #         x01,
    #         x02,
    #     )
    #     return np.concatenate((dx1_do, dx2_do))

    # def compareFiniteDiff_dX_dXo(self):

    #     self.setEllispoidsRadii()

    #     x0 = np.array([0, 1, 2, 6, 7, 8])

    #     J_finite_diff = self.numdiff(self.getXSol, x0)
    #     J = self.get_dY_dXo(x0)
    #     print(
    #         "--------------------------------- compareFiniteDiff_dX_dXo ----------------------"
    #     )
    #     print(f"finite diff : \n {J_finite_diff- J} \n")
    #     print(
    #         f"np.linalg.norm(J_finite_diff - J) : \n {np.linalg.norm(J_finite_diff - J)} \n"
    #     )

    # def get_gradient_xL_wrt_x0(self, x0):

    #     self.opt = DistOpt()
    #     self.opt.set_matrices()
    #     self.setEllispoidsRadii()
        
    #     self.opt.set_up_ellips()
    #     self.computeEllipsoidOpt(x0)

    #     self.opt.set_up_optim_var(
    #         self.x1, self.x2, self.distance, self.lambda1, self.lambda2
    #     )

    #     return self.opt.gradient_xL()

    # def get_gradient_xL_wrt_xSol(self, xSol):

    #     self.opt = DistOpt()
    #     self.opt.set_matrices()

    #     x1 = xSol[:3]
    #     x2 = xSol[3:]

    #     self.opt.set_up_ellips(self.A, self.B, self.x01, self.x02)

    #     # Initialize the QCQPSolver with the ellipsoid parameters
    #     qcqp_solver = EllipsoidOptimization()
    #     qcqp_solver.setup_problem(self.x01, self.A, self.x02, self.B)
    #     qcqp_solver.solve_problem(
    #         warm_start_primal=np.concatenate((self.x01, self.x02))
    #     )

    #     distance = qcqp_solver.get_minimum_cost()

    #     lambda1, lambda2 = qcqp_solver.get_dual_values()

    #     self.opt.set_up_optim_var(x1, x2, distance, lambda1, lambda2)

    #     return self.opt.gradient_xL()

    # def test_N(self):

    #     x0 = np.array([0, 1, 2, 6, 7, 8])

    #     self.setEllispoidsRadii()
    #     self.computeEllipsoidOpt(x0)

    #     self.opt.set_up_optim_var(
    #         self.x1, self.x2, self.distance, self.lambda1, self.lambda2
    #     )
    #     N = self.opt.N()

    #     ## Finite diff

    #     N_numdiff = self.numdiff(self.get_gradient_xL_wrt_x0, x0)

    #     print("--------------------------------- test_N ----------------------")
    #     print(f"N- N_numdiff :\n {N- N_numdiff}\n")
    #     print(f"np.linalg.norm(N- N_numdiff) :\n {np.linalg.norm(N- N_numdiff)}")

    #     print(f"N: \n {N[:6,:6]}")
    #     print(f"N_numdiff: \n {N_numdiff}")

    # def test_MN(self):
    #     MN = self.opt.invMN()
    #     print(MN.shape)

    # def test_M(self):

    #     x0 = np.array([0, 1, 2, 6, 7, 8])

    #     self.setEllispoidsRadii()
    #     self.opt = DistOpt()
    #     self.opt.set_matrices()

    #     self.x01 = x0[:3]
    #     self.x02 = x0[3:]
    #     self.opt.set_up_ellips(self.A, self.B, self.x01, self.x02)

    #     # Initialize the QCQPSolver with the ellipsoid parameters
    #     qcqp_solver = EllipsoidOptimization()
    #     qcqp_solver.setup_problem(self.x01, self.opt.A, self.x02, self.opt.B)
    #     qcqp_solver.solve_problem(
    #         warm_start_primal=np.concatenate((self.x01, self.x02))
    #     )

    #     x1, x2 = qcqp_solver.get_optimal_values()
    #     x = np.concatenate((x1, x2))
    #     distance = qcqp_solver.get_minimum_cost()

    #     lambda1, lambda2 = qcqp_solver.get_dual_values()

    #     self.opt.set_up_optim_var(x1, x2, distance, lambda1, lambda2)
    #     M = self.opt.M()

    #     ## Finite diff

    #     M_numdiff = self.numdiff(self.get_gradient_xL_wrt_xSol, x)

    #     print("--------------------------------- test_M ----------------------")
    #     print(f"M- M_numdiff : \n {M[:6,:6] - M_numdiff}")
    #     print(
    #         f"np.linalg.norm(M- M_numdiff) : \n {np.linalg.norm(M[:6,:6]- M_numdiff)}"
    #     )

    #     print(f"M: \n {M[:6,:6]}")
    #     print(f"M_numdiff: \n {M_numdiff}")

    # Numerical difference function
    def numdiff(self, f, inX, h=1e-6):
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

    # test = TestDistOpt()
    # test.setUp()
    # # # test.test_M()
    # # # test.test_N()
    # # # test.test_MN()
    # test.compareFiniteDiff_dX_dXo()
    # test.test_N()
    # test.test_M()

    # A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    # B = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
