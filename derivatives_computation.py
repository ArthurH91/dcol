import numpy as np
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
