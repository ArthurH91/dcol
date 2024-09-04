### This file defines a class used for the derivation of the movement of the closest points across the boundaries of the ellipsoids.

import numpy as np
import pinocchio as pin


class DerivativeComputation:
    """This class computes the derivative of the constraints with regards to the centers of the ellipsoids (center = c1, c2), the closest points (x = x1,x2) and the orientations of the matrices (R = R1, R2)."""

    def __init__(self) -> None:
        pass

    def lagrangian(self, x, center, A_1, A_2, lambda1, lambda2):
        x1 = x[:3]
        x2 = x[3:]
        c1 = center[:3]
        c2 = center[3:]
        print(np.linalg.norm(x1 - x2, 2))
        return (
            0.5 * np.linalg.norm(x1 - x2, 2)**2
            + lambda1 * ((x1 - c1).T @ A_1 @ (x1 - c1) - 1) / 2
            + lambda2 * ((x2 - c2).T @ A_2 @ (x2 - c2) - 1) / 2
        )
        
    def dl_dx(self, x, center, A_1, A_2, lambda1, lambda2):
        x1 = x[:3]
        x2 = x[3:]
        c1 = center[:3]
        c2 = center[3:] 
        
        dl = np.zeros(6)
        dl[:3] = (x1 - x2)  + lambda1 * A_1 @ (x1 - c1)       
        dl[3:] = -(x1 - x2) + lambda2 * A_2 @ (x2 - c2)

        return dl 
    
    def h1(self, x, center, A_1):
        x1 = x[:3]
        center_1 = center[:3]
        return ((x1 - center_1).T @ A_1 @ (x1 - center_1) - 1) / 2

    def h2(self, x, center, A_2):
        x2 = x[3:]
        center_2 = center[3:]
        return ((x2 - center_2).T @ A_2 @ (x2 - center_2) - 1) / 2

    def dh1_dx(self, x, center, A_1):
        dh1_dx_val = np.zeros((3, 2))
        x1 = x[:3]
        center_1 = center[:3]
        dh1_dx_val[:, 0] = A_1 @ (x1 - center_1)
        return dh1_dx_val

    def dh2_dx(self, x, center, A_2):
        dh2_dx_val = np.zeros((3, 2))
        x2 = x[3:]
        center_2 = center[3:]
        dh2_dx_val[:, 1] = A_2 @ (x2 - center_2)
        return dh2_dx_val

    def dh1_dcenter(self, x, center, A_1):
        dh1_dcenter_val = np.zeros((3, 2))
        x1 = x[:3]
        center_1 = center[:3]
        dh1_dcenter_val[:, 0] = -A_1 @ (x1 - center_1)
        return dh1_dcenter_val

    def dh2_dcenter(self, x, center, A_2):
        dh2_dcenter_val = np.zeros((3, 2))
        x2 = x[3:]
        center_2 = center[3:]
        dh2_dcenter_val[:, 1] = -A_2 @ (x2 - center_2)
        return dh2_dcenter_val

    def dh1_dR(self, x, center, A_1):
        dh1_dR_val = np.zeros((3, 2))
        x1 = x[:3]
        center_1 = center[:3]
        dh1_dR_val[:, 0] = (
            (1 / 2)
            * (x1 - center_1).T
            @ (-pin.skew(A_1 @ (x1 - center_1)) - A_1 @ pin.skew(x1 - center_1))
        )
        return dh1_dR_val

    def dh2_dR(self, x, center, A_2):
        dh2_dR_val = np.zeros((3, 2))
        x2 = x[:3]
        center_2 = center[:3]
        dh2_dR_val[:, 1] = (
            (1 / 2)
            * (x2 - center_2).T
            @ (-pin.skew(A_2 @ (x2 - center_2)) - A_2 @ pin.skew(x2 - center_2))
        )
        return dh2_dR_val
