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
        v1=np.zeros(3),
        w1=np.zeros(3),
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
        self.totalcost = (1 / 2) * casadi.sumsqr(self.x1)

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
        self.opti.solver("ipopt", {}, s_opt)

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
            Optimal value of x1.
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
        return (2 * self.opti.value(self.totalcost)) ** 0.5

    def get_lagrangian_value_at_opt(self):
        """
        Get the value of the Lagrangian function at the optimal solution.

        Returns:
            float: Value of the Lagrangian function at the optimal solution.
        """

        return self.opti.value(self.totalcost)

    def compute_x_at_opt(
        self,
        c1=np.ones(3),
        R1=np.eye((3)),
        radii1=np.ones(3),
        v1=np.zeros(3),
        w1=np.zeros(3),
    ):
        """ Compute the optimal value of x.

        Returns:
            np.array: Optimal value of x.
        """
        self.setup_problem(c1, R1, radii1, v1, w1)
        self.solve_problem()
        return self.get_optimal_values()

    def compute_distance_at_opt(
        self, c1=np.ones(3), R1=np.eye((3)), radii1=np.ones(3), v1=np.zeros(3), w1=np.zeros(3)
    ):
        """ Compute the distance between the closest points at the optimal solution.

        Returns:
            float: Distance between the closest points at the optimal solution.
        """
        self.setup_problem(c1, R1, radii1, v1, w1)
        self.solve_problem()
        return self.get_distance()  
        
    def compute_Lagrangian_at_opt(
        self,
        c1=np.ones(3),
        R1=np.eye((3)),
        radii1=np.ones(3),
        v1=np.zeros(3),
        w1=np.zeros(3),
    ):
        """ Compute the value of the Lagrangian function at the optimal solution.
        
        Returns:
            float: Value of the Lagrangian function at the optimal solution.
        """
        self.setup_problem(c1, R1, radii1, v1, w1)
        self.solve_problem()
        return self.get_lagrangian_value_at_opt()


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


def distance(radii1, radii2, c1, c2, R1, R2):

    radii1 = radii1
    radii2 = radii2
    c1 = c1
    c2 = c2
    R1 = R1
    R2 = R2

    ellipsoid1 = hppfcl.Ellipsoid(*radii1)
    ellipsoid2 = hppfcl.Ellipsoid(*radii2)

    ellipsoid1_pose = pin.SE3(rotation=R1, translation=np.array(c1))
    ellipsoid2_pose = pin.SE3(rotation=R2, translation=np.array(c2))

    request = hppfcl.DistanceRequest()
    request.gjk_max_iterations = 20000
    request.abs_err = 0
    request.gjk_tolerance = 1e-9
    result = hppfcl.DistanceResult()
    hppfcl_distance = hppfcl.distance(
        ellipsoid1,
        ellipsoid1_pose,
        ellipsoid2,
        ellipsoid2_pose,
        request,
        result,
    )
    return hppfcl_distance


def closest_points(radii1, radii2, c1, c2, R1, R2):

    radii1 = radii1
    radii2 = radii2
    c1 = c1
    c2 = c2
    R1 = R1
    R2 = R2

    ellipsoid1 = hppfcl.Ellipsoid(*radii1)
    ellipsoid2 = hppfcl.Ellipsoid(*radii2)

    ellipsoid1_pose = pin.SE3(rotation=R1, translation=np.array(c1))
    ellipsoid2_pose = pin.SE3(rotation=R2, translation=np.array(c2))

    request = hppfcl.DistanceRequest()
    request.gjk_max_iterations = 20000
    request.abs_err = 0
    request.gjk_tolerance = 1e-9
    result = hppfcl.DistanceResult()
    _ = hppfcl.distance(
        ellipsoid1,
        ellipsoid1_pose,
        ellipsoid2,
        ellipsoid2_pose,
        request,
        result,
    )
    closest_point_1 = result.getNearestPoint1()
    closest_point_2 = result.getNearestPoint2()
    return closest_point_1, closest_point_2


class TestEllipsoidDistance(unittest.TestCase):
    """
    Unit test class for EllipsoidOptimization.
    """

    def setUpWithoutRotation(self):
        """
        Set up the test environment.
        """

        self.R1 = pin.utils.rotate("z", 0.4701)
        self.R2 = np.eye(3)


        # Define initial positions for the centers of the two ellipsoids
        self.c1 = np.array([0.9, 1.6, 0])
        self.c2 = [0, 0, 0]

        # Speed of the ellipsoid 1
        v, w = np.r_[0.1, 0.2, 0], np.r_[0, 0, 0]

        # Define the radii for the ellipsoids
        self.radii_1 = [(1 / 14.13) ** 0.5, 1 / 5.34**0.5, 1]
        self.radii_2 = [1e-4, 1e-4, 1e-4]

        # Define the matrices representing the ellipsoids
        self.A = radii_to_matrix(self.radii_1)
        self.B = radii_to_matrix(self.radii_2)


        opt = EllipsoidOptimization()
        self.x = opt.compute_x_at_opt(self.c1, self.R1, self.radii_1, v, w)
        L = opt.compute_Lagrangian_at_opt(self.c1, self.R1, self.radii_1, v, w)
        self.d = opt.compute_distance_at_opt(self.c1, self.R1, self.radii_1, v, w)
        dt = 1e-6
        R1plus = pin.exp(w * dt) @ self.R1
        cplus = self.c1 + v * dt
        
        opt = EllipsoidOptimization()
        self.next_x = opt.compute_x_at_opt(cplus, R1plus, self.radii_1, v, w)
        next_L = opt.compute_Lagrangian_at_opt(cplus, R1plus, self.radii_1, v, w)
        
        self.Ldot_ND = (next_L - L) / dt
        self.Ldot = self.x @ v

    def setUpWithRotation(self):
        """
        Set up the test environment.
        """

        self.R1 = pin.utils.rotate("z", 0.4701)
        self.R2 = np.eye(3)


        # Define initial positions for the centers of the two ellipsoids
        self.c1 = np.array([0.9, 1.6, 0])
        self.c2 = [0, 0, 0]

        # Speed of the ellipsoid 1
        v, w = np.r_[0.1, 0.2, 0], np.r_[0, 0, 3.3]

        # Define the radii for the ellipsoids
        self.radii_1 = [(1 / 14.13) ** 0.5, 1 / 5.34**0.5, 1]
        self.radii_2 = [1e-4, 1e-4, 1e-4]

        # Define the matrices representing the ellipsoids
        self.A = radii_to_matrix(self.radii_1)
        self.B = radii_to_matrix(self.radii_2)


        opt = EllipsoidOptimization()
        self.x = opt.compute_x_at_opt(self.c1, self.R1, self.radii_1, v, w)
        L = opt.compute_Lagrangian_at_opt(self.c1, self.R1, self.radii_1, v, w)
        self.d = opt.compute_distance_at_opt(self.c1, self.R1, self.radii_1, v, w)
        dt = 1e-6
        R1plus = pin.exp(w * dt) @ self.R1
        cplus = self.c1 + v * dt
        
        opt = EllipsoidOptimization()
        self.next_x = opt.compute_x_at_opt(cplus, R1plus, self.radii_1, v, w)
        next_L = opt.compute_Lagrangian_at_opt(cplus, R1plus, self.radii_1, v, w)
        
        self.Ldot_ND = (next_L - L) / dt
        self.Ldot = self.x @ v

        self.Ldot_ND = (next_L - L) / dt
        Lr = np.cross(self.x, self.c1)
        Lc = self.x

        self.Ldot = Lc @ v + Lr @ w


    def test_compare_hppfcl_qcqp(self):
        """
        Compare the results from HPPFCL and QCQP solver.
        """
        self.setUpWithoutRotation()
        # Use HPPFCL to compute the distance and closest points between the two ellipsoids
        hppfcl_distance = distance(
            self.radii_1, self.radii_2, self.c1, self.c2, self.R1, self.R2
        )
        closest_point_1_hppfcl = closest_points(
            self.radii_1, self.radii_2, self.c1, self.c2, self.R1, self.R2
        )[0]
        # Compare the results from HPPFCL and QCQP
        self.assertAlmostEqual(
            hppfcl_distance, self.d, places=3, msg="Distances are not equal"
        )
        np.testing.assert_almost_equal(closest_point_1_hppfcl, self.x, decimal=5)

    def test_Ldot(self):
        """
        Test the derivative of the Lagrangian function with regards to time.
        """
        self.setUpWithoutRotation()
        self.assertAlmostEqual(
            np.linalg.norm(self.Ldot_ND - self.Ldot),
            0,
            places=4,
            msg="The value of the derivative of the Lagrangian function w.r.t. time is not equal to the finite different one.",
        )

    def test_Ldot_with_rotation(self):
        """
        Test the derivative of the Lagrangian function with regards to time.
        """
        self.setUpWithRotation()
        self.assertAlmostEqual(
            np.linalg.norm(self.Ldot_ND - self.Ldot),
            0,
            places=4,
            msg="The value of the derivative of the Lagrangian function w.r.t. time is not equal to the finite different one.",
        )


if __name__ == "__main__":
    unittest.main()
