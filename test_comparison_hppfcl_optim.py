import unittest
import numpy as np

import hppfcl
import pinocchio as pin

from ellipsoid_optimization import EllipsoidOptimization

class TestDistOpt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Define initial positions for the centers of the two ellipsoids
        cls.c1 = np.random.randn(3)
        cls.c2 = 10 * np.random.randn(3) + 10
        cls.center = np.concatenate((cls.c1, cls.c2))

        # Define the radii for the ellipsoids
        cls.radiiA = [1, 1, 1]
        cls.radiiB = [1, 1, 1]

        cls.shape1 = hppfcl.Ellipsoid(*cls.radiiA)
        cls.shape2 = hppfcl.Ellipsoid(*cls.radiiB)

        # Construct matrices A and B for ellipsoid constraints
        D1 = np.diag([1 / r**2 for r in cls.radiiA])
        D2 = np.diag([1 / r**2 for r in cls.radiiB])

        # Generate random rotation matrices using Pinocchio
        cls.R_A = pin.SE3.Random().rotation
        cls.R_B = pin.SE3.Random().rotation

        # Calculate rotated matrices
        cls.A1 = cls.R_A.T @ D1 @ cls.R_A
        cls.A2 = cls.R_B.T @ D2 @ cls.R_B

        cls.c1_SE3 = pin.SE3(rotation=cls.R_A.T, translation=cls.c1)
        cls.c2_SE3 = pin.SE3(rotation=cls.R_B.T, translation=cls.c2)

        cls.qcqp_solver = EllipsoidOptimization()
        cls.qcqp_solver.setup_problem(cls.c1, cls.A1, cls.c2, cls.A2, cls.R_A, cls.R_B)
        cls.qcqp_solver.solve_problem(warm_start_primal=np.concatenate((cls.c1, cls.c2)))

        cls.x1, cls.x2 = cls.qcqp_solver.get_optimal_values()
        cls.distance = cls.qcqp_solver.get_minimum_cost()
        cls.cp1, cls.cp2 = cp_hppfcl(cls.shape1, cls.c1_SE3, cls.shape2, cls.c2_SE3)
        
        cls.distance_v2 = cls.func_distance_annalytical(cls.center)
                
    @classmethod
    def func_distance_annalytical(cls, center):
        center_1 = center[:3]
        center_2 = center[3:]
        qcqp_solver = EllipsoidOptimization()
        qcqp_solver.setup_problem(center_1, cls.A1, center_2, cls.A2)
        qcqp_solver.solve_problem(warm_start_primal=center)
        x1, x2 = qcqp_solver.get_optimal_values()
        return np.linalg.norm(x1 - x2, 2)
    
    
    def test_dist(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(dist_hppfcl(cls.shape1, cls.c1_SE3, cls.shape2, cls.c2_SE3)- cls.distance_v2),
            0,
            places=6,
            msg = f"The distance computed from GJK ({dist_hppfcl(cls.shape1, cls.c1_SE3, cls.shape2, cls.c2_SE3)})is not the same as the distance computed with the QCQP ({cls.distance})."
        )
    
    def test_cp1(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(cls.cp1 - cls.x1),
            0,
            places=4,
            msg = f"The closest point 1 computed from GJK ({cls.cp1})is not the same as the one computed with the QCQP ({cls.x1})."
        )

    def test_cp2(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(cls.cp2 - cls.x2),
            0,
            places=4,
            msg = f"The closest point 2 computed from GJK ({cls.cp2})is not the same as the one computed with the QCQP ({cls.x2})."
        )

def dist_hppfcl(shape1, c1, shape2, c2):
    
    req = hppfcl.DistanceRequest()
    res = hppfcl.DistanceResult()
    
    return hppfcl.distance(shape1, c1, shape2, c2,req, res)

def cp_hppfcl(shape1, c1, shape2, c2):
    
    req = hppfcl.DistanceRequest()
    res = hppfcl.DistanceResult()
    
    _ =  hppfcl.distance(shape1, c1, shape2, c2,req, res)

    cp1 = res.getNearestPoint1()
    cp2 = res.getNearestPoint2()
    
    return cp1, cp2

if __name__ == "__main__":
    unittest.main()
