### This file defines a class used for the derivation of the movement of the closest points across the boundaries of the ellipsoids.

import numpy as np
import pinocchio as pin


class DerivativeComputation:
    """This class computes the derivative of the constraints with regards to the centers of the ellipsoids (center = c1, c2), the closest points (x = x1,x2) and the orientations of the matrices (R = R1, R2)."""

    def __init__(self) -> None:
        pass
    
    # def dist(self, center, A_1, A_2):
        
    
    def K(self, xlambda, center, A_1, A_2):
        """Compute the Karush-Kuhn-Tucker (KKT) conditions."""
        x = xlambda[:6]
        lambda1 = xlambda[6]
        lambda2 = xlambda[7]
        K = np.zeros(8)
        K[:6] = self.Lx(x, center, A_1, A_2, lambda1, lambda2)
        K[6] = self.h1(x, center, A_1)
        K[7] = self.h2(x, center, A_2)
        return K

    def M(self, xlambda, center, A_1, A_2):
        """Compute the Hessian of the Lagrangian function."""
        M = np.zeros((8, 8))
        x = xlambda[:6]
        lambda1 = xlambda[6]
        lambda2 = xlambda[7]
        M[:6, :6] = self.Lxx([lambda1, lambda2], A_1, A_2)
        M[6, :6] = self.dh1_dx(x, center, A_1)
        M[7, :6] = self.dh2_dx(x, center, A_2)
        M[:6, 6] = self.dh1_dx(x, center, A_1).T
        M[:6, 7] = self.dh2_dx(x, center, A_2).T
        return M
    
    def N(self, xlambda, center, A_1, A_2):
        """Compute the Jacobian of the constraints."""
        N = np.zeros((8, 6))
        x = xlambda[:6]
        N[6, :6] = self.dh1_dx(x, center, A_1)
        N[7, :6] = self.dh2_dx(x, center, A_2)
        return N

    def L(self, x, center, A_1, A_2, lambda1, lambda2):
        """Compute the Lagrangian function."""
        x1 = x[:3]
        x2 = x[3:]
        c1 = center[:3]
        c2 = center[3:]
        return (
            0.5 * np.linalg.norm(x1 - x2, 2)**2
            + lambda1 * ((x1 - c1).T @ A_1 @ (x1 - c1) - 1) / 2
            + lambda2 * ((x2 - c2).T @ A_2 @ (x2 - c2) - 1) / 2
        )
        
    def Lx(self, x, center, A_1, A_2, lambda1, lambda2):
        """Compute the derivative of the Lagrangian function with respect to x."""
        x1 = x[:3]
        x2 = x[3:]
        c1 = center[:3]
        c2 = center[3:] 
        
        dl = np.zeros(6)
        dl[:3] = (x1 - x2)  + lambda1 * A_1 @ (x1 - c1)       
        dl[3:] = -(x1 - x2) + lambda2 * A_2 @ (x2 - c2)

        return dl 
    
    def Lxx(self, lambda_, A_1, A_2):
        """Compute the Hessian of the Lagrangian function with respect to x."""
        lambda_1 = lambda_[0]
        lambda_2 = lambda_[1]

        H = np.zeros((6, 6))
        H[:3, 3:] = - np.eye(3)
        H[3:, :3] = - np.eye(3)
        H[:3, :3] =  np.eye(3) + lambda_1 * A_1
        H[3:, 3:] =  np.eye(3) + lambda_2 * A_2
        return H
    
    # def Lxeps(self, x, center, A_1, A_2, lambda1, lambda2):
        
    
    def h1(self, x, center, A_1):
        """Compute the first constraint."""
        x1 = x[:3]
        center_1 = center[:3]
        return ((x1 - center_1).T @ A_1 @ (x1 - center_1) - 1) / 2

    def h2(self, x, center, A_2):
        '''Compute the second constraint.'''
        x2 = x[3:]
        center_2 = center[3:]
        return ((x2 - center_2).T @ A_2 @ (x2 - center_2) - 1) / 2

    def dh1_dx(self, x, center, A_1):
        """Compute the derivative of the first constraint with respect to x."""
        dh1_dx_val = np.zeros(6)
        x1 = x[:3]
        center_1 = center[:3]
        dh1_dx_val[:3] = A_1 @ (x1 - center_1)
        return dh1_dx_val

    def dh2_dx(self, x, center, A_2):
        """Compute the derivative of the second constraint with respect to x."""
        dh2_dx_val = np.zeros(6)
        x2 = x[3:]
        center_2 = center[3:]
        dh2_dx_val[3:] = A_2 @ (x2 - center_2)
        return dh2_dx_val

    def dh1_dcenter(self, x, center, A_1):
        """Compute the derivative of the first constraint with respect to the center."""
        dh1_dcenter_val = np.zeros(6)
        x1 = x[:3]
        center_1 = center[:3]
        dh1_dcenter_val[:3] = -A_1 @ (x1 - center_1)
        return dh1_dcenter_val

    def dh2_dcenter(self, x, center, A_2):
        """Compute the derivative of the second constraint with respect to the center."""
        dh2_dcenter_val = np.zeros(6)
        x2 = x[3:]
        center_2 = center[3:]
        dh2_dcenter_val[3:] = -A_2 @ (x2 - center_2)
        return dh2_dcenter_val

    def dh1_dR(self, x, center, A_1):
        """Compute the derivative of the first constraint with respect to the orientation."""
        dh1_dR_val = np.zeros(6)
        x1 = x[:3]
        center_1 = center[:3]
        dh1_dR_val[:3] = (
            (1 / 2)
            * (x1 - center_1).T
            @ (-pin.skew(A_1 @ (x1 - center_1)) + A_1 @ pin.skew(x1 - center_1))
        )
        return dh1_dR_val

    def dh2_dR(self, x, center, A_2):
        """Compute the derivative of the second constraint with respect to the orientation."""
        dh2_dR_val = np.zeros(6)
        x2 = x[:3]
        center_2 = center[3:]
        dh2_dR_val[3:] = (
            (1 / 2)
            * (x2 - center_2).T
            @ (-pin.skew(A_2 @ (x2 - center_2)) + A_2 @ pin.skew(x2 - center_2))
        )
        return dh2_dR_val
