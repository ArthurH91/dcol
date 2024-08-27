import unittest
import numpy as np

import hppfcl
import pinocchio as pin

from ellipsoid_optimization import EllipsoidOptimization
from wrapper_panda import PandaWrapper
from distance_derivatives import dist, cp
from viewer import create_viewer, add_sphere_to_viewer

pin.seed(1)
np.random.seed(29)

class TestComparisonDistOpt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        ###! Definition of the shapes
        # Define initial positions for the centers of the two ellipsoids
        cls.c1_init = np.random.randn(3) 
        cls.c2_init = np.zeros(3)

        # Define initial rotation for the ellipsoids
        cls.R1_init = pin.SE3.Random().rotation
        cls.R2_init = pin.SE3.Random().rotation
        # Define the radii for the ellipsoids
        cls.radii1 = [0.1, 0.2, 0.1]
        cls.radii2 = [0.1, 0.1, 0.2]

        cls.shape1 = hppfcl.Ellipsoid(*cls.radii1)
        cls.shape2 = hppfcl.Ellipsoid(*cls.radii2)

        # Construct matrices A and B for ellipsoid constraints
        cls.D1 = np.diag([1 / r**2 for r in cls.radii1])
        cls.D2 = np.diag([1 / r**2 for r in cls.radii2])

        ###! Creation of the robot 
        # OBS CONSTANTS
        cls.PLACEMENT_OBS = pin.SE3(cls.R1_init, cls.c1_init)
        cls.DIM_OBS = cls.radii1

        # ELLIPS ON THE ROBOT
        cls.PLACEMENT_ROB = pin.SE3(cls.R2_init, cls.c2_init)
        cls.DIM_ROB = cls.radii2

        # Creating the robot
        robot_wrapper = PandaWrapper()
        cls.rmodel, cls.cmodel, cls.vmodel = robot_wrapper()

        cls.cmodel = robot_wrapper.add_2_ellips(
            cls.cmodel,
            placement_obs=cls.PLACEMENT_OBS,
            dim_obs=cls.DIM_OBS,
            placement_rob=cls.PLACEMENT_ROB,
            dim_rob=cls.DIM_ROB,
        )
        # Creating the data models
        cls.rdata = cls.rmodel.createData()
        cls.cdata = cls.cmodel.createData()

        cls.viz = create_viewer(cls.rmodel, cls.cmodel, cls.vmodel)

        
    @classmethod
    def update_A_R_c(cls, q,v = None):
        
        pin.updateGeometryPlacements(cls.rmodel, cls.rdata, cls.cmodel, cls.cdata, q)
        pin.framesForwardKinematics(cls.rmodel, cls.rdata, q)
        if v is not None:
            pin.forwardKinematics(cls.rmodel, cls.rdata, q, v)
        
        R1 = cls.cdata.oMg[cls.cmodel.getGeometryId("obstacle")].rotation
        R2 = cls.cdata.oMg[cls.cmodel.getGeometryId("ellips_rob")].rotation

        c1 = cls.cdata.oMg[cls.cmodel.getGeometryId("obstacle")].translation
        c2 = cls.cdata.oMg[cls.cmodel.getGeometryId("ellips_rob")].translation

        # # Calculate rotated matrices
        A1 = R1.T @ cls.D1 @ R1
        A2 = R2.T @ cls.D2 @ R2
        
        return A1, A2, R1, R2, c1, c2
    
    @classmethod
    def setup_random_config(cls):
        
        q = pin.randomConfiguration(cls.rmodel)
        v = pin.randomConfiguration(cls.rmodel)
        x = np.concatenate((q, v))

        A1, A2, R1, R2, c1, c2 = cls.update_A_R_c(q, v)
        
        c1_SE3 = pin.SE3(R1.T, c1)
        c2_SE3 = pin.SE3(R2.T, c2)

        A1_SE3 = pin.SE3(A1.T, c1)
        A2_SE3 = pin.SE3(A2.T, c2)

        cls.cp1_hppfcl_without_robot, cls.cp2_hppfcl_without_robot = cp_hppfcl(cls.shape1, c1_SE3, cls.shape2, c2_SE3)
        cls.dist_hppfcl_without_robot = dist_hppfcl(cls.shape1, c1_SE3, cls.shape2, c2_SE3)
        cls.dist_opt_val = dist_opt(cls.shape1, c1_SE3, cls.shape2, c2_SE3)
        cls.cp1_opt_val, cls.cp2_opt_val = cp_opt(cls.shape1, c1_SE3, cls.shape2, c2_SE3)

        # ###! Vizualisation
        add_sphere_to_viewer(cls.viz, "cp1_hppfcl_without_robot", 1e-2, cls.cp1_hppfcl_without_robot)
        add_sphere_to_viewer(cls.viz, "cp2_hppfcl_without_robot", 1e-2, cls.cp2_hppfcl_without_robot)

        add_sphere_to_viewer(cls.viz, "cp1_opt", 2e-2, cls.cp1_opt_val)
        add_sphere_to_viewer(cls.viz, "cp2_opt", 2e-2, cls.cp2_opt_val)

        cls.viz.display(q)        

    def test_cp1_random_config(cls):
        cls.setup_random_config()

        cls.assertAlmostEqual(
            np.linalg.norm(cls.cp1_hppfcl_without_robot - cls.cp1_opt_val),
            0,
            places=9,
            msg=f"The closest point 1 computed from GJK ({cls.cp1_hppfcl_without_robot})is not the same as the one computed with the QCQP ({cls.cp1_opt_val}).",
        )
        
    def test_cp2_random_config(cls):
        cls.setup_random_config()
        cls.assertAlmostEqual(
            np.linalg.norm(cls.cp2_hppfcl_without_robot - cls.cp2_opt_val),
            0,
            places=9,
            msg=f"The closest point 2 computed from GJK ({cls.cp2_hppfcl_without_robot})is not the same as the one computed with the QCQP ({cls.cp2_opt_val}).",
        )

    def test_dist_random_config(cls):
        
        cls.setup_random_config()
        cls.assertAlmostEqual(
            np.linalg.norm(
                 cls.dist_hppfcl_without_robot
                - cls.dist_opt_val
            ),
            0,
            places=5,
            msg=f"The distance computed from GJK ({cls.dist_hppfcl_without_robot})is not the same as the distance computed with the QCQP ({cls.dist_opt_val}).",
        )

def dist_hppfcl(shape1, c1_se3, shape2, c2_se3):

    req = hppfcl.DistanceRequest()
    res = hppfcl.DistanceResult()
    req.gjk_max_iterations = 20000
    req.abs_err = 0
    req.gjk_tolerance = 1e-9
    return hppfcl.distance(shape1, c1_se3, shape2, c2_se3, req, res)


def cp_hppfcl(shape1, c1_se3, shape2, c2_se3):

    req = hppfcl.DistanceRequest()
    res = hppfcl.DistanceResult()
    req.gjk_max_iterations = 20000
    req.abs_err = 0
    req.gjk_tolerance = 1e-9
    _ = hppfcl.distance(shape1, c1_se3, shape2, c2_se3, req, res)
    cp1 = res.getNearestPoint1()
    cp2 = res.getNearestPoint2()
    return cp1, cp2

def cp_opt(shape1, c1_se3, shape2, c2_se3):
    
    D1 = np.diag([1 / r**2 for r in shape1.radii])
    D2 = np.diag([1 / r**2 for r in shape2.radii])
    
    R1 = c1_se3.rotation
    R2 = c2_se3.rotation

    A1 = R1.T @ D1 @ R1
    A2 = R2.T @ D2 @ R2

    qcqp_solver = EllipsoidOptimization()
    qcqp_solver.setup_problem(c1_se3.translation, A1, c2_se3.translation, A2)
    qcqp_solver.solve_problem(
        warm_start_primal=np.concatenate((c1_se3.translation, c2_se3.translation))
    )

    cp1, cp2 = qcqp_solver.get_optimal_values()
    return cp1,cp2

def dist_opt(shape1, c1_se3, shape2, c2_se3):
    x1,x2 = cp_opt(shape1, c1_se3, shape2, c2_se3)
    return np.linalg.norm(x1-x2,2)


if __name__ == "__main__":
    unittest.main()
