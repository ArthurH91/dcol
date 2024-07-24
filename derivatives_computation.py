import unittest
import numpy as np
from qcqp_solver import EllipsoidOptimization
import pinocchio as pin

import hppfcl


class DerivativeComputation:
    """This class takes into input the center of the two ellipsoids and their radii and returns the derivative of the closest points with regards to the movement of the centers."""

    def __init__(self) -> None:
        pass

    def lagrangian(self, x, lambda_, center, A, B):
        x1 = x[:3]
        x2 = x[3:]
        lambda_1 = lambda_[0]
        lambda_2 = lambda_[1]
        center_1 = center[:3]
        center_2 = center[3:]

        return (
            np.linalg.norm(x1 - x2, 2)
            + lambda_1 * ((x1 - center_1).T @ A @ (x1 - center_1) - 1) / 2
            + lambda_2 * ((x2 - center_2).T @ B @ (x2 - center_2) - 1) / 2
        ).item()

    def h1(self, x, center, A):
        x1 = x[:3]
        center_1 = center[:3]
        return (((x1 - center_1).T @ A @ (x1 - center_1) - 1) / 2).item()

    def h2(self, x, center, B):
        x2 = x[3:]
        center_2 = center[3:]
        return (((x2 - center_2).T @ B @ (x2 - center_2) - 1) / 2).item()

    def dh1_dx(self, x, center, A):
        x1 = x[:3]
        center_1 = center[:3]
        g1 = A @ (x1 - center_1)
        g2 = np.zeros(3)
        return np.concatenate((g1, g2))

    def dh1_dcenter(self, x, center, A):
        x1 = x[:3]
        center_1 = center[:3]
        g1 = -A @ (x1 - center_1)
        g2 = np.zeros(3)
        return np.concatenate((g1, g2))

    def dh2_dcenter(self, x, center, B):
        x2 = x[3:]
        center_2 = center[3:]
        g1 = np.zeros(3)
        g2 = -B @ (x2 - center_2)
        return np.concatenate((g1, g2))

    def dh2_dx(self, x, center, B):
        x2 = x[3:]
        center_2 = center[3:]
        g1 = np.zeros(3)
        g2 = B @ (x2 - center_2)
        return np.concatenate((g1, g2))

    def grad_x(self, x, lambda_, center, A, B):
        x1 = x[:3]
        x2 = x[3:]
        lambda_1 = lambda_[0]
        lambda_2 = lambda_[1]
        center_1 = center[:3]
        center_2 = center[3:]
        g1 = (x1 - x2) / np.linalg.norm(x1 - x2, 2) + lambda_1 * A @ (x1 - center_1)
        g2 = -(x1 - x2) / np.linalg.norm(x1 - x2, 2) + lambda_2 * B @ (x2 - center_2)
        return np.concatenate((g1, g2))

    def hessian_xx(self, x, lambda_, center, A, B):
        x1 = x[:3]
        x2 = x[3:]
        lambda_1 = lambda_[0]
        lambda_2 = lambda_[1]

        H = np.zeros((6, 6))

        d = np.linalg.norm(x1 - x2, 2)

        v = (x1 - x2).reshape((3, 1)) @ (x1 - x2).reshape((1, 3))

        I_diag = np.eye(3) / d - v / (d**2) ** (3 / 2)
        I_off_diag = -np.eye(3) / d + v / (d**2) ** (3 / 2)

        H[:3, 3:] = I_off_diag
        H[3:, :3] = I_off_diag
        H[:3, :3] = I_diag + lambda_1 * A
        H[3:, 3:] = I_diag + lambda_2 * B

        return H

    def hessian_center_x(self, x, lambda_, center, A, B):
        lambda_1 = lambda_[0]
        lambda_2 = lambda_[1]

        H = np.zeros((6, 6))
        H[:3, :3] = -lambda_1 * A
        H[3:, 3:] = -lambda_2 * B

        return H

    def compute_dx_dcenter_hppfcl(self, shape1, shape2, placement1, placement2, A, B):


        center = np.concatenate((placement1.translation, placement2.translation))

        x = self.compute_closest_points_hppfcl(shape1, shape2, placement1, placement2)
        lambda_ = self.compute_lambda_hppfcl(shape1, shape2, placement1, placement2, A, B)

        M_matrix = np.zeros((8, 8))
        N_matrix = np.zeros((8, 6))

        dh1_dx_ = self.dh1_dx(x, center, A)
        dh2_dx_ = self.dh2_dx(x, center, B)

        M_matrix[:6, :6] = self.hessian_xx(x, lambda_, center, A, B)
        M_matrix[:6, 6] = dh1_dx_
        M_matrix[:6, 7] = dh2_dx_
        M_matrix[6, :6] = dh1_dx_.T
        M_matrix[7, :6] = dh2_dx_.T

        dh1_do_ = self.dh1_dcenter(x, center, A)
        dh2_do_ = self.dh2_dcenter(x, center, B)

        N_matrix[:6, :] = self.hessian_center_x(x, lambda_, center, A, B)
        N_matrix[6, :] = dh1_do_
        N_matrix[7, :] = dh2_do_

        dy = -np.linalg.solve(M_matrix, N_matrix)

        return dy[:6]

    def compute_distance_hppfcl(self, shape1, shape2, placement1, placement2):

        # Compute distances and nearest points using HPP-FCL
        req = hppfcl.DistanceRequest()
        req.gjk_max_iterations = 20000
        req.abs_err = 0
        req.gjk_tolerance = 1e-9
        res = hppfcl.DistanceResult()
        dist = hppfcl.distance(shape1, placement1, shape2, placement2, req, res)
        return dist

    def compute_closest_points_hppfcl(self, shape1, shape2, placement1, placement2):

        # Compute distances and nearest points using HPP-FCL
        req = hppfcl.DistanceRequest()
        req.gjk_max_iterations = 20000
        req.abs_err = 0
        req.gjk_tolerance = 1e-9
        res = hppfcl.DistanceResult()
        _ = hppfcl.distance(shape1, placement1, shape2, placement2, req, res)
        cp1 = res.getNearestPoint1()
        cp2 = res.getNearestPoint2()
        return np.concatenate((cp1, cp2))

    def compute_lambda_hppfcl(self, shape1, shape2, placement1, placement2, A, B):

        center_1 = placement1.translation
        center_2 = placement2.translation

        x = self.compute_closest_points_hppfcl(shape1, shape2, placement1, placement2)
        x1 = x[:3]
        x2 = x[3:]

        l1 = -np.linalg.norm(x1 - x2, 2) / ((x1 - x2).T @ A @ (x1 - center_1)).item()
        l2 = np.linalg.norm(x1 - x2, 2) / ((x1 - x2).T @ B @ (x2 - center_2)).item()
        return np.array([l1, l2])


class TestDistOpt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Define initial positions for the centers of the two ellipsoids
        cls.x0_1 = np.random.randn(3)
        cls.x0_2 = 10 * np.random.randn(3) + 10
        cls.center = np.concatenate((cls.x0_1, cls.x0_2))

        # Define the radii for the ellipsoids
        cls.radiiA = [1, 2, 1]
        cls.radiiB = [1, 2, 1]

        cls.shape1 = hppfcl.Ellipsoid(*cls.radiiA)
        cls.shape2 = hppfcl.Ellipsoid(*cls.radiiB)

        # Construct matrices A and B for ellipsoid constraints
        A_ = np.diag([1 / r**2 for r in cls.radiiA])
        B_ = np.diag([1 / r**2 for r in cls.radiiB])

        # Generate random rotation matrices using Pinocchio
        cls.R_A = pin.SE3.Random().rotation
        cls.R_B = pin.SE3.Random().rotation

        # Calculate rotated matrices
        cls.A = cls.R_A.T @ A_ @ cls.R_A
        cls.B = cls.R_B.T @ B_ @ cls.R_B

        cls.x = np.random.random(6)
        cls.lambda_ = np.random.random(2)

        # Define initial positions for the centers of the two ellipsoids
        x0_1 = np.random.randn(3)
        x0_2 = 10 * np.random.randn(3) + 10
        cls.center = np.concatenate((x0_1, x0_2))

        cls.centerA = pin.SE3(rotation=cls.R_A.T, translation=x0_1)
        cls.centerB = pin.SE3(rotation=cls.R_B.T, translation=x0_2)

        cls.derivativeComputation = DerivativeComputation()

        cls.grad_x_ND = numdiff(
            lambda variable: cls.derivativeComputation.lagrangian(
                variable, cls.lambda_, cls.center, cls.A, cls.B
            ),
            cls.x,
        )
        cls.hessian_center_x_ND = numdiff(
            lambda variable: cls.derivativeComputation.grad_x(
                cls.x, cls.lambda_, variable, cls.A, cls.B
            ),
            cls.center,
        )
        cls.hessian_xx_ND = numdiff(
            lambda variable: cls.derivativeComputation.grad_x(
                variable, cls.lambda_, cls.center, cls.A, cls.B
            ),
            cls.x,
        )
        cls.dh1_dx_ND = numdiff(
            lambda variable: cls.derivativeComputation.h1(variable, cls.center, cls.A),
            cls.x,
        )
        cls.dh2_dx_ND = numdiff(
            lambda variable: cls.derivativeComputation.h2(variable, cls.center, cls.B),
            cls.x,
        )
        cls.dh1_dcenter_ND = numdiff(
            lambda variable: cls.derivativeComputation.h1(cls.x, variable, cls.A),
            cls.center,
        )
        cls.dh2_dcenter_ND = numdiff(
            lambda variable: cls.derivativeComputation.h2(cls.x, variable, cls.B),
            cls.center,
        )
        cls.dx_dcenter_ND = numdiff(lambda variable: cls.x_star(variable), cls.center)

    @classmethod
    def func_lambda_annalytical(cls, center):
        center_1 = center[:3]
        center_2 = center[3:]
        qcqp_solver = EllipsoidOptimization()
        qcqp_solver.setup_problem(center_1, cls.A, center_2, cls.B)
        qcqp_solver.solve_problem(warm_start_primal=center)
        x1, x2 = qcqp_solver.get_optimal_values()

        l1 = (
            -np.linalg.norm(x1 - x2, 2) / ((x1 - x2).T @ cls.A @ (x1 - center_1)).item()
        )
        l2 = np.linalg.norm(x1 - x2, 2) / ((x1 - x2).T @ cls.B @ (x2 - center_2)).item()
        return np.array([l1, l2])

    @classmethod
    def func_distance_annalytical(cls, center):
        center_1 = center[:3]
        center_2 = center[3:]
        qcqp_solver = EllipsoidOptimization()
        qcqp_solver.setup_problem(center_1, cls.A, center_2, cls.B)
        qcqp_solver.solve_problem(warm_start_primal=center)
        x1, x2 = qcqp_solver.get_optimal_values()

        return np.linalg.norm(x1 - x2, 2)

    @classmethod
    def x_star(cls, center):
        center_1 = center[:3]
        center_2 = center[3:]
        qcqp_solver = EllipsoidOptimization()
        qcqp_solver.setup_problem(center_1, cls.A, center_2, cls.B)
        qcqp_solver.solve_problem(warm_start_primal=center)
        xSol1, xSol2 = qcqp_solver.get_optimal_values()
        return np.concatenate((xSol1, xSol2))

    @classmethod
    def func_lambda(cls, center):
        center_1 = center[:3]
        center_2 = center[3:]
        qcqp_solver = EllipsoidOptimization()
        qcqp_solver.setup_problem(center_1, cls.A, center_2, cls.B)
        qcqp_solver.solve_problem(warm_start_primal=center)

        lambda1, lambda2 = qcqp_solver.get_dual_values()

        return np.array([lambda1, lambda2])

    @classmethod
    def func_distance(cls, center):
        center_1 = center[:3]
        center_2 = center[3:]
        qcqp_solver = EllipsoidOptimization()
        qcqp_solver.setup_problem(center_1, cls.A, center_2, cls.B)
        qcqp_solver.solve_problem(warm_start_primal=center)
        return qcqp_solver.get_minimum_cost()

    @classmethod
    def dx_dcenter(cls, center):
        center_1 = center[:3]
        center_2 = center[3:]
        qcqp_solver = EllipsoidOptimization()
        qcqp_solver.setup_problem(center_1, cls.A, center_2, cls.B)
        qcqp_solver.solve_problem(warm_start_primal=center)
        xSol1, xSol2 = qcqp_solver.get_optimal_values()
        lambda1, lambda2 = qcqp_solver.get_dual_values()

        x = np.concatenate((xSol1, xSol2))
        lambda_ = np.array([lambda1, lambda2])

        M_matrix = np.zeros((8, 8))
        N_matrix = np.zeros((8, 6))

        dh1_dx_ = cls.derivativeComputation.dh1_dx(x, center, cls.A)
        dh2_dx_ = cls.derivativeComputation.dh2_dx(x, center, cls.B)

        M_matrix[:6, :6] = cls.derivativeComputation.hessian_xx(
            x, lambda_, center, cls.A, cls.B
        )
        M_matrix[:6, 6] = dh1_dx_
        M_matrix[:6, 7] = dh2_dx_
        M_matrix[6, :6] = dh1_dx_.T
        M_matrix[7, :6] = dh2_dx_.T

        dh1_do_ = cls.derivativeComputation.dh1_dcenter(x, center, cls.A)
        dh2_do_ = cls.derivativeComputation.dh2_dcenter(x, center, cls.B)

        N_matrix[:6, :] = cls.derivativeComputation.hessian_center_x(
            x, lambda_, center, cls.A, cls.B
        )
        N_matrix[6, :] = dh1_do_
        N_matrix[7, :] = dh2_do_

        dy = -np.linalg.solve(M_matrix, N_matrix)

        return dy[:6]

    ### TESTING THE ANALYTICAL WITH THE VALUES FROM NUMDIFF

    def test_gradient_x_lagrangian(cls):

        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.grad_x_ND
                - cls.derivativeComputation.grad_x(
                    cls.x, cls.lambda_, cls.center, cls.A, cls.B
                )
            ),
            0,
            places=5,
            msg="The value of the gradient of the lagrangian w.r.t. the closest points is not equal to the finite different one.",
        )

    def test_hessian_center_x(cls):

        cls.assertAlmostEqual(
            (
                np.linalg.norm(
                    cls.hessian_center_x_ND
                    - cls.derivativeComputation.hessian_center_x(
                        cls.x, cls.lambda_, cls.center, cls.A, cls.B
                    )
                )
            ),
            0,
            places=5,
            msg="The value of the hessian of the lagrangian w.r.t. the closest points and of the centers of the ellipsoids is not equal to the finite different one.",
        )

    def test_hessian_xx(cls):

        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.hessian_xx_ND
                - cls.derivativeComputation.hessian_xx(
                    cls.x, cls.lambda_, cls.center, cls.A, cls.B
                )
            ),
            0,
            places=4,
            msg="The value of the hessian of the lagrangian w.r.t. the closest points is not equal to the finite different one.",
        )

    def test_dh1_dx(cls):

        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh1_dx_ND
                - cls.derivativeComputation.dh1_dx(cls.x, cls.center, cls.A)
            ),
            0,
            places=5,
            msg="The value of the derivative of the hard constraint h1 w.r.t. the closest points is not equal to the finite different one.",
        )

    def test_dh2_dx(cls):

        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh2_dx_ND
                - cls.derivativeComputation.dh2_dx(cls.x, cls.center, cls.B)
            ),
            0,
            places=5,
            msg="The value of the derivative of the hard constraint h2 w.r.t. the closest points is not equal to the finite different one.",
        )

    def test_dh1_dcenter(cls):

        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh1_dcenter_ND
                - cls.derivativeComputation.dh1_dcenter(cls.x, cls.center, cls.A)
            ),
            0,
            places=5,
            msg="The value of the derivative of the hard constraint h1 w.r.t. the center of the ellipsoids is not equal to the finite different one.",
        )

    def test_dh2_dcenter(cls):

        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dh1_dcenter_ND
                - cls.derivativeComputation.dh1_dcenter(cls.x, cls.center, cls.A)
            ),
            0,
            places=5,
            msg="The value of the derivative of the hard constraint h2 w.r.t. the center of the ellipsoids is not equal to the finite different one.",
        )

    #### TESTING THE VALUES FOUND BY ANALYTICAL RESULTS AND OPTIMISATION
    def test_lambdas(cls):

        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.func_lambda_annalytical(cls.center) - cls.func_lambda(cls.center)
            ),
            0,
            places=5,
            msg="The value of the lambdas are not equal to the one found in the optimization problem.",
        )

    def test_distance(cls):

        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.func_distance_annalytical(cls.center)
                - cls.func_distance(cls.center)
            ),
            0,
            places=5,
            msg="The value of the distance is not equal to the one found in the optimization problem.",
        )

    def test_dx_dcenter(cls):

        cls.assertAlmostEqual(
            np.linalg.norm(cls.dx_dcenter_ND - cls.dx_dcenter(cls.center)),
            0,
            places=2,
            msg="The value of the matrix is not equal to the one found in the optimization problem.",
        )

    ##### TESTING THE RESULTS FOUND BY HPPFCL AND THE RESULTS FROM OPTIMISATION

    def test_distance_hppfcl(cls):

        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.func_distance_annalytical(cls.center)
                - cls.derivativeComputation.compute_distance_hppfcl(
                    cls.shape1, cls.shape2, cls.centerA, cls.centerB
                )
            ),
            0,
            places=4,
            msg="The value of the distance is not equal to the one found in the optimization problem.",
        )

    def test_closest_points_hppfcl(cls):

        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.derivativeComputation.compute_closest_points_hppfcl(
                    cls.shape1, cls.shape2, cls.centerA, cls.centerB
                )
                - cls.x_star(cls.center)
            ),
            0,
            places=3,
            msg="The value of the closest points is not equal to the one found in the optimization problem.",
        )

    def test_lambda_hppfcl(cls):

        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.func_lambda_annalytical(cls.center)
                - cls.derivativeComputation.compute_lambda_hppfcl(
                    cls.shape1, cls.shape2, cls.centerA, cls.centerB, cls.A, cls.B
                )
            ),
            0,
            places=3,
            msg="The value of the lambdas are not equal to the one found in hppfcl.",
        )

    def test_dx_dcenter_hppfcl(cls):

        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.dx_dcenter_ND
                - cls.derivativeComputation.compute_dx_dcenter_hppfcl(
                    cls.shape1, cls.shape2, cls.centerA, cls.centerB, cls.A, cls.B
                )
            ),
            0,
            places=2,
            msg="The value of the matrix is not equal to the one found in the optimization problem.",
        )
        
    


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
