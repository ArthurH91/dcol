import unittest
import numpy as np

import hppfcl
import pinocchio as pin

from ellipsoid_optimization import radii_to_matrix, EllipsoidOptimization

class TestEllipsoidDistance(unittest.TestCase):
    """
    Unit test class for EllipsoidOptimization.
    """

    def setUp(self):
        """
        Set up the test environment.
        """
        # Define initial positions for the centers of the two ellipsoids
        self.x0_1 = [4, 0, 3]
        self.x0_2 = [0, 0, 0]

        # Define the radii for the ellipsoids
        self.radii_1 = [1.5, 1, 1]
        self.radii_2 = [2, 1, 1.9]

        # Define the matrices representing the ellipsoids
        self.A = radii_to_matrix(self.radii_1)
        self.B = radii_to_matrix(self.radii_2)

        # Add some rotation
        self.A_rot = pin.utils.rotate("x", np.pi/4)
        self.B_rot = pin.utils.rotate("y", np.pi/2)
        # self.A_rot = np.eye(3)
        # self.B_rot = np.eye(3)
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
        self.ellipsoid_2 = hppfcl.Ellipsoid(
            self.radii_2[0], self.radii_2[1], self.radii_2[2]
        )
        self.ellipsoid_1_pose = pin.SE3(rotation=self.A_rot, translation= np.array(self.x0_1))
        self.ellipsoid_2_pose = pin.SE3(rotation=self.B_rot, translation= np.array(self.x0_2))
        

    def test_qcqp_solver(self):
        """
        Test the QCQP solver to ensure it provides valid results.
        """
        self.assertIsNotNone(self.x1, "x1 should not be None")
        self.assertIsNotNone(self.x2, "x2 should not be None")
        self.assertGreaterEqual(
            self.distance, 0, "Minimum absolute difference should be non-negative"
        )


    def test_compare_hppfcl_qcqp(self):
        """
        Compare the results from HPPFCL and QCQP solver.
        """
        # Use HPPFCL to compute the distance and closest points between the two ellipsoids
        request = hppfcl.DistanceRequest()
        request.gjk_max_iterations = 20000
        request.abs_err = 0
        request.gjk_tolerance = 1e-9
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
        self.assertAlmostEqual(
            hppfcl_distance, self.distance, places=4, msg="Distances are not equal"
        )
        np.testing.assert_almost_equal(closest_point_1_hppfcl, self.x1, decimal=5)
        np.testing.assert_almost_equal(closest_point_2_hppfcl, self.x2, decimal=5)



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
