### This file is to verify the that the analitical values of the dual variables are the right one, compared with CasADi.
import pinocchio as pin
import contextlib
import os
import numpy as np
import casadi
from distance_derivatives import dist, cp, h1, h2, R, A

class EllipsoidOptimization:
    """
    Class for setting up and solving an optimization problem for ellipsoids using CasADi.
    """

    def __init__(self, rmodel, cmodel, shape1_name, shape2_name, ellipsoid_dim=3):
        """
        Initialize the EllipsoidOptimization class.

        Args:
            ellipsoid_dim (int, optional): Dimension of the ellipsoids. Defaults to 3.
        """
        self.opti = casadi.Opti()
        self.totalcost = None
        self.solution = None
        self.ellipsoid_dim = ellipsoid_dim

        self.rmodel = rmodel
        self.cmodel = cmodel
        self.rdata = rmodel.createData()
        self.cdata = cmodel.createData()

        self.shape1_name = shape1_name
        self.shape2_name = shape2_name
        self.shape1_id = cmodel.getGeometryId(shape1_name)
        self.shape2_id = cmodel.getGeometryId(shape2_name)

        self.x1 = self.opti.variable(ellipsoid_dim)
        self.x2 = self.opti.variable(ellipsoid_dim)

    def setup_problem(
        self, 
        q,
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

        pin.forwardKinematics(self.rmodel, self.rdata, q)
        pin.updateGeometryPlacements(self.rmodel, self.rdata, self.cmodel, self.cdata)        
        self.c1 = self.cdata.oMg[self.shape1_id].translation # Center of the shapes
        self.c2 = self.cdata.oMg[self.shape2_id].translation # Center of the shapes

        self.A1 = A(self.rmodel, self.cmodel, q, self.shape1_name)
        self.A2 = A(self.rmodel, self.cmodel, q, self.shape2_name)

        # Define the cost function (distance between closest points)
        self.totalcost = (1/2)*casadi.sumsqr((self.x1 - self.x2)) 

        # Define the constraints for the ellipsoids
        self.con1 = (self.x1 - self.c1).T @ self.A1 @ (self.x1 - self.c1) / 2 == 1 / 2
        self.opti.subject_to(self.con1)
        self.con2 = (self.x2 - self.c2).T @ self.A2 @ (self.x2 - self.c2) / 2 == 1 / 2
        self.opti.subject_to(self.con2)

    def solve_problem(self, warm_start_primal=None):
        """
        Solve the optimization problem.

        Args:
            warm_start_primal (np.ndarray, optional): Initial guess for the solver. Defaults to None.
        """
        self.opti.solver('ipopt')

        self.opti.minimize(self.totalcost)

        if warm_start_primal is None:
            warm_start_primal = np.concatenate([self.c1, self.c2])
        # Apply warm start values if provided
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

