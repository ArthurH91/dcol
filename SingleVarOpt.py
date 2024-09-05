### This file is to verify the that the analitical values of the dual variables are the right one, compared with CasADi.
import unittest
import contextlib
import os
import hppfcl
import numpy as np
import casadi
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
        self.totalcost = None
        self.solution = None

    def setup_problem(
        self,
        c1=np.ones(3),
        R1=np.eye((3)),
        radii1=np.ones(3),
        v1 = np.zeros(3),
        w1 = np.zeros(3),
    ):
        """
        Set up the optimization problem.

        Args:
            x0_1 (np.ndarray, optional): Center of the first ellipsoid. Defaults to np.ones(3).
            A (np.ndarray, optional): Shape matrix of the first ellipsoid. Defaults to np.array([np.ones((3, 3))]).
            x0_2 (np.ndarray, optional): Center of the second ellipsoid. Defaults to 3*np.ones(3).
            B (np.ndarray, optional): Shape matrix of the second ellipsoid. Defaults to np.array([np.ones((3, 3))]).
        """
        
        self.c1 = c1
        self.R1 = R1
        self.D1 = radii_to_matrix(radii1)
        self.A1 = self.R1 @ self.D1 @ self.R1.T
        self.v1 = v1
        self.w1 = w1
        
        # Define the cost function (distance between closest points)
        self.totalcost = (1 / 2) * casadi.sumsqr(self.x1 )

        # Define the constraints for the ellipsoids
        self.con1 = (self.x1 - c1).T @ self.A1 @ (self.x1 - c1) / 2 == 1 / 2
        self.opti.subject_to(self.con1)
        
    def solve_problem(self, warm_start_primal=None):
        """
        Solve the optimization problem.

        Args:
            warm_start_primal (np.ndarray, optional): Initial guess for the solver. Defaults to None.
        """
        s_opt = {
            "tol": 1e-4,
            "acceptable_tol": 1e-4,
            "max_iter": 300,
        }
        self.opti.solver("ipopt", {},s_opt)

        self.opti.minimize(self.totalcost)

        # Apply warm start values if provided
        if warm_start_primal is not None:
            self.opti.set_initial(self.x1, warm_start_primal[: self.ellipsoid_dim])

        try:
            with open(os.devnull, "w") as fnull:
                with contextlib.redirect_stdout(fnull):
                    self.solution = self.opti.solve()
        except RuntimeError as e:
            print(f"Solver failed: {e}")
            # Print current values of variables for debugging
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
        return x1_sol

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
        return con1_dual
    
    def get_distance(self):
        """
        Get the distance between the closest points.

        Returns:
            float: Distance between the closest points.
        """
        return (2 * self.opti.value(self.totalcost))**0.5


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
    
    
class TestEllipsoidDistance(unittest.TestCase):
    """
    Unit test class for EllipsoidOptimization.
    """

    def setUp(self):
        """
        Set up the test environment.
        """

        
        R = pin.utils.rotate('z',.4701)
        D = np.diagflat([14.13,5.34,1])
        A, c = R@D@R.T, np.array([ .9,1.6,0 ]) 
        v,w = np.r_[.1,.2,0], np.r_[0,0,0]

        # Define initial positions for the centers of the two ellipsoids
        self.x0_1 = c
        self.x0_2 = [0, 0, 0]

        # Define the radii for the ellipsoids
        self.radii_1 = [(1/14.13)**0.5,1/5.34**0.5,1]
        self.radii_2 = [1e-4, 1e-4, 1e-4]

        # Define the matrices representing the ellipsoids
        self.A = radii_to_matrix(self.radii_1)
        self.B = radii_to_matrix(self.radii_2)

        # Add some rotation
        self.R1 = R
        self.R2 = np.eye(3)
        
        # Create HPPFCL ellipsoid objects and their corresponding collision objects
        self.ellipsoid_1 = hppfcl.Ellipsoid(
            self.radii_1[0], self.radii_1[1], self.radii_1[2]
        )
        self.ellipsoid_2 = hppfcl.Ellipsoid(
            self.radii_2[0], self.radii_2[1], self.radii_2[2]
        )
        self.ellipsoid_1_pose = pin.SE3(rotation=self.R1, translation= np.array(self.x0_1))
        self.ellipsoid_2_pose = pin.SE3(rotation=self.R2, translation= np.array(self.x0_2))
        

        opt = EllipsoidOptimization()
        opt.setup_problem(c, self.R1, self.radii_1, v, w)
        opt.solve_problem()
        self.sol_x = opt.get_optimal_values()
        sol_lam = opt.get_dual_values()
        sol_f = opt.get_minimum_cost()
        self.sol_d = opt.get_distance()
        sol_L = sol_f
        
        dt = 1e-6
        Rplus = pin.exp(w*dt)@R
        cplus = c+v*dt
        Aplus = Rplus@D@Rplus.T
        
        
        opt = EllipsoidOptimization()
        opt.setup_problem(cplus, Rplus, self.radii_1, v, w)
        opt.solve_problem()
        next_sol_x = opt.get_optimal_values()
        next_sol_lam = opt.get_dual_values()
        next_sol_f = opt.get_minimum_cost()
        next_sol_d = opt.get_distance()
        next_sol_L = next_sol_f
        
            
        self.Ldot_ND = (next_sol_L-sol_L)/dt
        self.Ldot = self.sol_x@v

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
        # Compare the results from HPPFCL and QCQP
        self.assertAlmostEqual(
            hppfcl_distance, self.sol_d, places=3, msg="Distances are not equal"
        )
        np.testing.assert_almost_equal(closest_point_1_hppfcl, self.sol_x, decimal=5)

    def test_Ldot(self):
        """
        Test the derivative of the Lagrangian function with regards to time.
        """
        
        self.assertAlmostEqual(
            np.linalg.norm(
                self.Ldot_ND
                - self.Ldot
            ),
            0,
            places=4,
            msg="The value of the derivative of the Lagrangian function w.r.t. time is not equal to the finite different one.",
        )
        
if __name__ == "__main__":
    unittest.main()

