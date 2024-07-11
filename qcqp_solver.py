import unittest
import numpy as np
from scipy.optimize import minimize
import hppfcl
import pinocchio as pin

class QCQPSolver:
    def __init__(self, x0_1, A, x0_2, B, R_A=None, R_B=None):
        # Initialize the QCQP solver with the centers and matrices of two ellipsoids
        self.x0_1 = np.array(x0_1)
        self.x0_2 = np.array(x0_2)
        self.A = np.array(A)
        self.B = np.array(B)
        
        # Use identity matrices if rotation matrices are not provided
        self.R_A = np.eye(3) if R_A is None else np.array(R_A)
        self.R_B = np.eye(3) if R_B is None else np.array(R_B)
        
        # Calculate rotated matrices
        self.A_rot = self.R_A.T @ self.A @ self.R_A
        self.B_rot = self.R_B.T @ self.B @ self.R_B

    def objective(self, x):
        # Define the objective function to minimize the L1 norm of the difference between two points
        x1, x2 = x[:3], x[3:]
        return np.linalg.norm(x1 - x2, 2)
    
    def constraint1(self, x):
        # Define the constraint for the first ellipsoid
        x1 = x[:3]
        return np.dot((x1 - self.x0_1).T, np.dot(self.A_rot, (x1 - self.x0_1))) - 1

    def constraint2(self, x):
        # Define the constraint for the second ellipsoid
        x2 = x[3:]
        return np.dot((x2 - self.x0_2).T, np.dot(self.B_rot, (x2 - self.x0_2))) - 1

    def solve(self, trust = False):
        # Initial guess for the optimization
        x0 = np.hstack([self.x0_1, self.x0_2])
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': self.constraint1},
            {'type': 'eq', 'fun': self.constraint2}
        ]
        
        # Solve the optimization problem
        if trust:
            result = minimize(self.objective, x0, constraints=constraints, method="trust-constr" ,options = {"verbose": 3, "maxiter": 100000, "xtol":1e-6, "gtol": 1e-6})
        else:
            result = minimize(self.objective, x0, constraints=constraints)

        # Extract the resulting points
        x1, x2 = result.x[:3], result.x[3:]
        if trust:
            return x1, x2, result.fun, result.v
        
        return x1, x2, result.fun
    
class TestEllipsoidDistance(unittest.TestCase):

    def setUp(self):
        
        self.solved = False
        # Define initial positions for the centers of the two ellipsoids
        self.x0_1 = [4, 0, 0]
        self.x0_2 = [0, 0, 1]
        
        # Define the radii for the ellipsoids
        self.radii_1 = [1.5, 1, 1]
        self.radii_2 = [2, 1, 1.9]
        
        
        # Define the matrices representing the ellipsoids
        self.A = radii_to_matrix(self.radii_1)
        self.B = radii_to_matrix(self.radii_2)
        
        # Initialize the QCQPSolver with the ellipsoid parameters
        self.qcqp_solver = QCQPSolver(self.x0_1, self.A, self.x0_2, self.B)
        
        
        # Create HPPFCL ellipsoid objects and their corresponding collision objects
        self.ellipsoid_1 = hppfcl.Ellipsoid(self.radii_1[0], self.radii_1[1], self.radii_1[2])
        self.ellipsoid_1_pose = pin.SE3.Identity()
        self.ellipsoid_1_pose.translation = np.array(self.x0_1)
        self.ellipsoid_2 = hppfcl.Ellipsoid(self.radii_2[0], self.radii_2[1], self.radii_2[2])
        self.ellipsoid_2_pose = pin.SE3.Identity()
        self.ellipsoid_2_pose.translation = np.array(self.x0_2)

    def ttest_qcqp_solver(self):
        # Solve the QCQP problem and print results
        if not self.solved:
            trust = True
            if trust:
                self.x1, self.x2, self.min_abs_diff,self.lagrange_multipliers = self.qcqp_solver.solve(trust=trust)
            else:
                self.x1, self.x2, self.min_abs_diff= self.qcqp_solver.solve()    
            self.solved = True
        # Check if the results are valid
        self.assertIsNotNone(self.x1, "x1 should not be None")
        self.assertIsNotNone(self.x2, "x2 should not be None")
        self.assertGreaterEqual(self.min_abs_diff, 0, "Minimum absolute difference should be non-negative")

    def test_compare_lagrangian(self):
        
        if not self.solved:
            self.x1, self.x2, self.min_abs_diff,self.lagrange_multipliers = self.qcqp_solver.solve(trust=True)
            self.solved = True
        lambda1 = self.min_abs_diff / (np.dot((self.x1 - self.x2).T, np.dot(self.A, (self.x1 - self.x2))))
        print(f"lagragian multipliers: {self.lagrange_multipliers}")
        np.testing.assert_almost_equal(lambda1, self.lagrange_multipliers[0], decimal=3)

        

    def ttest_compare_hppfcl_qcqp(self):
        
        trust = True
        # Use HPPFCL to compute the distance and closest points between the two ellipsoids
        request = hppfcl.DistanceRequest()
        result = hppfcl.DistanceResult()
        hppfcl_distance = hppfcl.distance(self.ellipsoid_1, self.ellipsoid_1_pose, self.ellipsoid_2, self.ellipsoid_2_pose, request, result)
        
        closest_point_1_hppfcl = result.getNearestPoint1()
        closest_point_2_hppfcl = result.getNearestPoint2()
        
        # Solve the QCQP problem to get the closest points
        if trust:
            self.x1_qcqp, self.x2_qcqp, self.qcqp_distance, self.lagrange_multipliers = self.qcqp_solver.solve(trust=trust)
        else:
            self.x1_qcqp, self.x2_qcqp, self.qcqp_distance= self.qcqp_solver.solve()
        
        print("HPPFCL Results:")
        print("Closest Point on Ellipsoid 1:", closest_point_1_hppfcl)
        print("Closest Point on Ellipsoid 2:", closest_point_2_hppfcl)
        
        print("QCQP Solver Results:")
        print("Optimal x1:", self.x1_qcqp)
        print("Optimal x2:", self.x2_qcqp)
        if trust:
            print(f"Lagrange multipliers: {self.lagrange_multipliers}")
        # Compare the results from HPPFCL and QCQP
        self.assertAlmostEqual(hppfcl_distance, self.qcqp_distance,places=4, msg="Distances are not equal")
        np.testing.assert_almost_equal(closest_point_1_hppfcl, self.x1_qcqp, decimal=3)
        np.testing.assert_almost_equal(closest_point_2_hppfcl, self.x2_qcqp, decimal=3)

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
    return np.array([
        [1/radii[0]**2, 0, 0],
        [0, 1/radii[1]**2, 0],
        [0, 0, 1/radii[2]**2]
    ])
    
class TestRadiiToMatrix(unittest.TestCase):

    def test_radii_to_matrix(self):
        # Test with radii 1, 2, 3
        radii = 1, 2, 3
        expected_matrix = np.array([
            [1/1**2, 0, 0],
            [0, 1/2**2, 0],
            [0, 0, 1/3**2]
        ])
        result_matrix = radii_to_matrix(radii)
        np.testing.assert_array_almost_equal(result_matrix, expected_matrix, decimal=6)

        # Test with radii 2, 3, 4
        radii = 2, 3, 4
        expected_matrix = np.array([
            [1/2**2, 0, 0],
            [0, 1/3**2, 0],
            [0, 0, 1/4**2]
        ])
        result_matrix = radii_to_matrix(radii)
        np.testing.assert_array_almost_equal(result_matrix, expected_matrix, decimal=6)

        # Test with radii 5, 5, 5 (sphere)
        radii = 5, 5, 5
        expected_matrix = np.array([
            [1/5**2, 0, 0],
            [0, 1/5**2, 0],
            [0, 0, 1/5**2]
        ])
        result_matrix = radii_to_matrix(radii)
        np.testing.assert_array_almost_equal(result_matrix, expected_matrix, decimal=6)


if __name__ == '__main__':
    unittest.main()
