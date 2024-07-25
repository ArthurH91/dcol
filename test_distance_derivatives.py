import unittest
import numpy as np

import hppfcl
import pinocchio as pin

from wrapper_panda import PandaWrapper
from distance_derivatives import dist, ddist_dt


class TestDistOpt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
         # OBS CONSTANTS
        cls.PLACEMENT_OBS = pin.SE3(pin.utils.rotate("x", 0), np.array([0, 0, 2]))
        cls.DIM_OBS = [0.1, 0.1, 0.4]

        # ELLIPS ON THE ROBOT
        cls.PLACEMENT_ROB = pin.SE3(pin.utils.rotate("x", 0), np.array([0, 0, 0]))
        cls.DIM_ROB = [0.2, 0.1, 0.2]

        # Creating the robot
        robot_wrapper = PandaWrapper()
        cls.rmodel, cls.cmodel, _ = robot_wrapper()

        cls.cmodel = robot_wrapper.add_ellips(
            cls.cmodel,
            placement_obs=cls.PLACEMENT_OBS,
            dim_obs=cls.DIM_OBS,
            placement_rob=cls.PLACEMENT_ROB,
            dim_rob=cls.DIM_ROB,
        )
        cls.q = pin.randomConfiguration(cls.rmodel)
        cls.v = pin.randomConfiguration(cls.rmodel)

        cls.ddist_dt_ND = finite_diff_time(cls.q,cls.v,
            lambda variable: dist(
                cls.rmodel, cls.cmodel, variable,),
        )
        
    def test_ddist_dt(cls):
        
        cls.assertAlmostEqual(
            np.linalg.norm(cls.ddist_dt_ND - ddist_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)))), 0, places=2, msg="The time derivative of the distance is not equal to the one from numdiff"
        )


    

def finite_diff_time(q, v,f, h=1e-6):
    return (f(q + h * v) - f(q)) / h


def numdiff(f, q, h=1e-6):
    j_diff = np.zeros(len(q))
    fx = f(q)
    for i in range(len(q)):
        e = np.zeros(len(q))
        e[i] = h
        j_diff[i] = ((f(q + e) - fx) / e[i])
    return j_diff

def numdiff_matrix(f, q, h=1e-6):
    fx = f(q)
    j_diff = np.zeros((len(q), len(fx)))
    for i in range(len(q)):
        e = np.zeros(len(q))
        e[i] = h
        j_diff[i,:] = ((f(q + e) - fx) / e[i]).reshape((6,))
    return j_diff

if __name__ == "__main__":
    unittest.main()