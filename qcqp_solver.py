# ellipsoid_optimization.py
import contextlib
import os
import unittest
import numpy as np
from scipy.optimize import minimize
import casadi

import hppfcl
import pinocchio as pin


class EllipsoidOptimization:
    """
    Class for setting up and solving an optimization problem for ellipsoids using CasADi.
    """

    def __init__(self, ellipsoid_dim=3):
        """
        Initialize the EllipsoidOptimization class.

        Args:
            ellipsoid_dim (int, optional): Dimension of the ellipsoids. Defaults to 3.
        """
        self.ellipsoid_dim = ellipsoid_dim
        self.opti = casadi.Opti()
        self.x1 = self.opti.variable(self.ellipsoid_dim)
        self.x2 = self.opti.variable(self.ellipsoid_dim)
        self.totalcost = None
        self.solution = None

    def setup_problem(
        self,
        x0_1=np.ones(3),
        A=np.array([np.ones((3, 3))]),
        x0_2=3 * np.ones(3),
        B=np.array([np.ones((3, 3))]),
        R_A=None,
        R_B=None,
    ):
        """
        Set up the optimization problem.

        Args:
            x0_1 (np.ndarray, optional): Center of the first ellipsoid. Defaults to np.ones(3).
            A (np.ndarray, optional): Shape matrix of the first ellipsoid. Defaults to np.array([np.ones((3, 3))]).
            x0_2 (np.ndarray, optional): Center of the second ellipsoid. Defaults to 3*np.ones(3).
            B (np.ndarray, optional): Shape matrix of the second ellipsoid. Defaults to np.array([np.ones((3, 3))]).
            R_A (np.ndarray, optional): Rotation matrix for the first ellipsoid. Defaults to None.
            R_B (np.ndarray, optional): Rotation matrix for the second ellipsoid. Defaults to None.
        """
        # Use identity matrices if rotation matrices are not provided
        self.R_A = np.eye(self.ellipsoid_dim) if R_A is None else np.array(R_A)
        self.R_B = np.eye(self.ellipsoid_dim) if R_B is None else np.array(R_B)

        # Calculate rotated matrices
        self.A_rot = self.R_A.T @ A @ self.R_A
        self.B_rot = self.R_B.T @ B @ self.R_B

        # Define the cost function (distance between closest points)
        self.totalcost = casadi.norm_2((self.x1 - self.x2))**2 / 2

        # Define the constraints for the ellipsoids
        self.con1 = (self.x1 - x0_1).T @ self.A_rot @ (self.x1 - x0_1) / 2 == 1 / 2
        self.opti.subject_to(self.con1)
        self.con2 = (self.x2 - x0_2).T @ self.B_rot @ (self.x2 - x0_2) / 2 == 1 / 2
        self.opti.subject_to(self.con2)

    def solve_problem(self, warm_start_primal=None):
        """
        Solve the optimization problem.

        Args:
            warm_start_primal (np.ndarray, optional): Initial guess for the solver. Defaults to None.
        """
        self.opti.solver('ipopt')

        self.opti.minimize(self.totalcost)

        # Apply warm start values if provided
        if warm_start_primal is not None:
            self.opti.set_initial(self.x1, warm_start_primal[: self.ellipsoid_dim])
            self.opti.set_initial(self.x2, warm_start_primal[self.ellipsoid_dim:])

        try:
            with open(os.devnull, 'w') as fnull:
                with contextlib.redirect_stdout(fnull):  
                    self.solution = self.opti.solve()
        except RuntimeError as e:
            print(f"Solver failed: {e}")
            # Print current values of variables for debugging
            print("p2:", self.opti.debug.value(self.x2))
            print("p1:", self.opti.debug.value(self.x1))
            print("totalcost:", self.opti.debug.value(self.totalcost))
            raise

    def get_optimal_values(self):
        """
        Get the optimal values of the decision variables.

        Returns:
            tuple: Optimal values of x1 and x2.
        """
        x1_sol = self.opti.value(self.x1)
        x2_sol = self.opti.value(self.x2)
        return x1_sol, x2_sol

    def get_minimum_cost(self):
        """
        Get the minimum cost value.

        Returns:
            float: Minimum cost value.
        """
        return self.opti.value(self.totalcost)

    def get_dual_values(self):
        """
        Get the dual values for the constraints.

        Returns:
            tuple: Dual values for con1 and con2.
        """
        con1_dual = self.opti.value(self.opti.dual(self.con1))
        con2_dual = self.opti.value(self.opti.dual(self.con2))
        return con1_dual, con2_dual


class TestEllipsoidDistance(unittest.TestCase):
    """
    Unit test class for EllipsoidOptimization.
    """

    def setUp(self):
        """
        Set up the test environment.
        """
        # Define initial positions for the centers of the two ellipsoids
        self.x0_1 = [4, 0, 0]
        self.x0_2 = [0, 0, 1]

        # Define the radii for the ellipsoids
        self.radii_1 = [1.5, 1, 1]
        self.radii_2 = [2, 1, 1.9]

        # Define the matrices representing the ellipsoids
        self.A = radii_to_matrix(self.radii_1)
        self.B = radii_to_matrix(self.radii_2)

        # Add some rotation
        self.A_rot = pin.utils.rotate("x", np.pi/4)
        self.B_rot = pin.utils.rotate("y", np.pi/2)

        # Initialize the QCQPSolver with the ellipsoid parameters
        self.qcqp_solver = EllipsoidOptimization()
        self.qcqp_solver.setup_problem(self.x0_1, self.A, self.x0_2, self.B, self.A_rot, self.B_rot)
        self.qcqp_solver.solve_problem(warm_start_primal=np.concatenate((self.x0_1, self.x0_2)))

        self.x1, self.x2 = self.qcqp_solver.get_optimal_values()
        self.distance = self.qcqp_solver.get_minimum_cost()

        # Create HPPFCL ellipsoid objects and their corresponding collision objects
        self.ellipsoid_1 = hppfcl.Ellipsoid(
            self.radii_1[0], self.radii_1[1], self.radii_1[2]
        )
        self.ellipsoid_1_pose = pin.SE3.Identity()
        self.ellipsoid_1_pose.translation = np.array(self.x0_1)
        self.ellipsoid_1_pose.rotation = self.A_rot
        self.ellipsoid_2 = hppfcl.Ellipsoid(
            self.radii_2[0], self.radii_2[1], self.radii_2[2]
        )
        self.ellipsoid_2_pose = pin.SE3.Identity()
        self.ellipsoid_2_pose.translation = np.array(self.x0_2)
        self.ellipsoid_2_pose.rotation = self.B_rot

    def lagrangian_multipliers(
        self, x01: np.ndarray, A: np.ndarray, x02: np.ndarray, B: np.ndarray
    ):
        """
        Calculate the Lagrangian multipliers for the ellipsoids.

        Args:
            x01 (np.ndarray): Center of the first ellipsoid.
            A (np.ndarray): Shape matrix of the first ellipsoid.
            x02 (np.ndarray): Center of the second ellipsoid.
            B (np.ndarray): Shape matrix of the second ellipsoid.

        Returns:
            tuple: Lagrangian multipliers for the constraints.
        """
        lambda1 = -self.distance / (
            np.dot(np.dot((self.x1 - self.x2).T, A), (self.x1 - x01))
        )
        lambda2 = self.distance / (
            np.dot(np.dot((self.x1 - self.x2).T, B), (self.x2 - x02))
        )

        return lambda1, lambda2

    def test_qcqp_solver(self):
        """
        Test the QCQP solver to ensure it provides valid results.
        """
        self.assertIsNotNone(self.x1, "x1 should not be None")
        self.assertIsNotNone(self.x2, "x2 should not be None")
        self.assertGreaterEqual(
            self.distance, 0, "Minimum absolute difference should be non-negative"
        )

    def test_compare_lagrangian(self):
        """
        Compare the Lagrangian multipliers from the CasADi solver and manual calculation.
        """
        self.lagrange_multipliers_casadi = self.qcqp_solver.get_dual_values()

        lambda1, lambda2 = self.lagrangian_multipliers(
            self.x0_1, self.qcqp_solver.A_rot, self.x0_2, self.qcqp_solver.B_rot
        )
        # np.testing.assert_almost_equal(
        #     lambda1, self.lagrange_multipliers_casadi[0], decimal=3
        # )
        # np.testing.assert_almost_equal(
        #     lambda2, self.lagrange_multipliers_casadi[1], decimal=3
        # )

    def test_compare_hppfcl_qcqp(self):
        """
        Compare the results from HPPFCL and QCQP solver.
        """
        # Use HPPFCL to compute the distance and closest points between the two ellipsoids
        request = hppfcl.DistanceRequest()
        result = hppfcl.DistanceResult()
        hppfcl_distance = hppfcl.distance(
            self.ellipsoid_1,
            self.ellipsoid_1_pose,
            self.ellipsoid_2,
            self.ellipsoid_2_pose,
            request,
            result,
        )

        closest_point_1_hppfcl = result.getNearestPoint1()
        closest_point_2_hppfcl = result.getNearestPoint2()

        # Compare the results from HPPFCL and QCQP
        # self.assertAlmostEqual(
        #     hppfcl_distance, self.distance, places=4, msg="Distances are not equal"
        # )
        np.testing.assert_almost_equal(closest_point_1_hppfcl, self.x1, decimal=3)
        np.testing.assert_almost_equal(closest_point_2_hppfcl, self.x2, decimal=3)


def radii_to_matrix(radii):
    """
    Converts ellipsoid radii to the matrix representation.

    Parameters:
    a (float): Radius along the x-axis.
    b (float): Radius along the y-axis.
    c (float): Radius along the z-axis.

    Returns:
    numpy.ndarray: 3x3 matrix representation of the ellipsoid.
    """
    return np.array(
        [
            [1 / radii[0] ** 2, 0, 0],
            [0, 1 / radii[1] ** 2, 0],
            [0, 0, 1 / radii[2] ** 2],
        ]
    )


class TestRadiiToMatrix(unittest.TestCase):
    """
    Unit test class for radii_to_matrix function.
    """

    def test_radii_to_matrix(self):
        """
        Test the radii_to_matrix function with different sets of radii.
        """
        # Test with radii 1, 2, 3
        radii = 1, 2, 3
        expected_matrix = np.array(
            [[1 / 1**2, 0, 0], [0, 1 / 2**2, 0], [0, 0, 1 / 3**2]]
        )
        result_matrix = radii_to_matrix(radii)
        np.testing.assert_array_almost_equal(result_matrix, expected_matrix, decimal=6)

        # Test with radii 2, 3, 4
        radii = 2, 3, 4
        expected_matrix = np.array(
            [[1 / 2**2, 0, 0], [0, 1 / 3**2, 0], [0, 0, 1 / 4**2]]
        )
        result_matrix = radii_to_matrix(radii)
        np.testing.assert_array_almost_equal(result_matrix, expected_matrix, decimal=6)

        # Test with radii 5, 5, 5 (sphere)
        radii = 5, 5, 5
        expected_matrix = np.array(
            [[1 / 5**2, 0, 0], [0, 1 / 5**2, 0], [0, 0, 1 / 5**2]]
        )
        result_matrix = radii_to_matrix(radii)
        np.testing.assert_array_almost_equal(result_matrix, expected_matrix, decimal=6)


if __name__ == "__main__":
    unittest.main()
