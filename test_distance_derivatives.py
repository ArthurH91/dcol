import unittest
import numpy as np

import pinocchio as pin

from wrapper_panda import PandaWrapper
from distance_derivatives import dist,ddist_dq, ddist_dt, cp, dX_dq, dddist_dt_dq, h1, h2, R, dR_dt, A, dA_dt
from viewer import create_viewer, add_sphere_to_viewer

pin.seed(np.random.randint(0, 1000))

class TestDistOpt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        def R1(rmodel, cmodel, x):
            return R(rmodel, cmodel, x, shape_name="obstacle")
        def R2(rmodel, cmodel, x):
            return R(rmodel, cmodel, x, shape_name="ellips_rob")
        def A1(rmodel, cmodel, x):
            return A(rmodel, cmodel, x, shape_name="obstacle")
        def A2(rmodel, cmodel, x):
            return A(rmodel, cmodel, x, shape_name="ellips_rob")

        # OBS CONSTANTS
        cls.PLACEMENT_OBS = pin.SE3(pin.utils.rotate("x", 0), np.array([0, 0, 2]))
        # cls.DIM_OBS = [0.2, .2, .2] # Sphere
        cls.DIM_OBS = [0.2, .16, .1] # Ellipsoid

        # ELLIPS ON THE ROBOT
        cls.PLACEMENT_ROB = pin.SE3(pin.utils.rotate("x", 0), np.array([0, 0, 0]))
        cls.DIM_ROB = [.1, .2, .3] # Ellipsoid
        # cls.DIM_ROB = [10.0, 10.0, 10.0] # Sphere
        
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
        # cls.q = pin.randomConfiguration(cls.rmodel)
        cls.q = pin.neutral(cls.rmodel)
        # cls.q[0] = 1
        # cls.v = pin.randomConfiguration(cls.rmodel)
        cls.v = np.random.random(cls.rmodel.nv)
        cls.x = np.concatenate((cls.q, cls.v))

        cls.ddist_dt_ND = finite_diff_time(
            cls.q,
            cls.v,
            lambda variable: dist(
                cls.rmodel,
                cls.cmodel,
                variable,
            ),
        )
        
        cls.R1_dot_ND = finite_diff_time(
            cls.q, 
            cls.v,
            lambda variable: R1(
                cls.rmodel,
                cls.cmodel,
                variable
            )
        )
        cls.R2_dot_ND = finite_diff_time(
            cls.q, 
            cls.v,
            lambda variable: R2(
                cls.rmodel,
                cls.cmodel,
                variable
            )
        )

        cls.A1_dot_ND = finite_diff_time(
            cls.q, 
            cls.v,
            lambda variable: A1(
                cls.rmodel,
                cls.cmodel,
                variable
            )
        )
        cls.A2_dot_ND = finite_diff_time(
            cls.q, 
            cls.v,
            lambda variable: A2(
                cls.rmodel,
                cls.cmodel,
                variable
            )
        )

        cls.cp = cp(cls.rmodel, cls.cmodel, cls.q)
        cls.dx_dq_ND = finite_difference_jacobian(
            lambda variable: cp(cls.rmodel, cls.cmodel, variable), cls.q
        )
        # cls.dddist_dt_dq_ND = numdiff(
        #     lambda variable: ddist_dt(cls.rmodel, cls.cmodel, variable), cls.x
        # )[:7]
        
        cls.ddist_dq_ND = numdiff(
            lambda variable: dist(cls.rmodel, cls.cmodel, variable), cls.q
        )
        
        cls.dh1_dq_ND = numdiff(
            lambda variable: h1(cls.rmodel, cls.cmodel,np.array([0, 0, 2]), cls.cp[:3] ,variable), cls.q
        )
        cls.dh2_dq_ND = numdiff(
            lambda variable: h2(cls.rmodel,cls.cmodel,np.array([0, 0, 2]),cls.cp[3:] ,variable), cls.q
        )

        viz = create_viewer(cls.rmodel, cls.cmodel, cls.vmodel)
        viz.display(cls.q)


    def test_ddist_dt(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.ddist_dt_ND
                - ddist_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)))
            ),
            0,
            places=4,
            msg=f"The time derivative of the distance is not equal to the one from numdiff. \n The value of the numdiff is : \n {cls.ddist_dt_ND}\n and the value computed is : \n {ddist_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)))}",
        )
        
    def test_dR1_dt(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.R1_dot_ND
                - dR_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), shape_name="obstacle")
            ),
            0,
            places=4,
            msg=f"The time derivative of the rotation matrix is not equal to the one from numdiff. \n The value of the numdiff is : \n {cls.R1_dot_ND}\n and the value computed is : \n {dR_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), shape_name="obstacle")}",
        )
        
    def test_dR2_dt(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.R2_dot_ND
                - dR_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), shape_name="ellips_rob")
            ),
            0,
            places=4,
            msg=f"The time derivative of the rotation matrix is not equal to the one from numdiff. \n The value of the numdiff is : \n {cls.R2_dot_ND}\n and the value computed is : \n {dR_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), shape_name="ellips_rob")}",
        )
        
    def test_dA1_dt(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.A1_dot_ND
                - dA_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), shape_name="obstacle")
            ),
            0,
            places=4,
            msg=f"The time derivative of the rotation matrices is not equal to the one from numdiff. \n The value of the numdiff is : \n {cls.A1_dot_ND}\n and the value computed is : \n {dA_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), shape_name="obstacle")}",
        )
        
    def test_dA2_dt(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.A2_dot_ND
                - dA_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), shape_name="ellips_rob")
            ),
            0,
            places=4,
            msg=f"The time derivative of the rotation matrices is not equal to the one from numdiff. \n The value of the numdiff is : \n {cls.A2_dot_ND}\n and the value computed is : \n {dA_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), shape_name="ellips_rob")}",
        )

    # def test_ddist_dq(cls):

    #     cls.assertAlmostEqual(
    #         np.linalg.norm(
    #             cls.ddist_dq_ND
    #             - ddist_dq(cls.rmodel, cls.cmodel,cls.q)
    #         ),
    #         0,
    #         places=4,
    #         msg=f"The time derivative of the distance is not equal to the one from numdiff. \n The value of the numdiff is : \n {cls.ddist_dq_ND}\n and the value computed is : \n {ddist_dq(cls.rmodel, cls.cmodel,cls.q)}",
    #     )
    # def test_dcp1_dq(cls):

    #     cls.assertAlmostEqual(
    #         np.linalg.norm(
    #             cls.dx_dq_ND.T[:3] - dX_dq(cls.rmodel, cls.cmodel, cls.q)[:3]
    #         ),
    #         0,
    #         places=2,
    #         msg=f"The derivative of the closest point 1 w.r.t q is not equal to the one from numdiff. \n The value of the numdiff is : \n {cls.dx_dq_ND.T[:3]}\n and the value computed is : \n {dX_dq(cls.rmodel, cls.cmodel, cls.q)[:3]}",
    #     )

    # def test_dcp2_dq(cls):

    #     cls.assertAlmostEqual(
    #         np.linalg.norm(
    #             cls.dx_dq_ND.T[3:] - dX_dq(cls.rmodel, cls.cmodel, cls.q)[3:]
    #         ),
    #         0,
    #         places=2,
    #         msg=f"The derivative of the closest point 2 w.r.t q is not equal to the one from numdiff. \n The value of the numdiff is : \n {cls.dx_dq_ND.T[3:]}\n and the value computed is : \n {dX_dq(cls.rmodel, cls.cmodel, cls.q)[3:]}",
    # #     )

    # def test_dddist_dt_dq(cls):

        
    #     cls.assertAlmostEqual(
    #         np.linalg.norm(
    #             cls.dddist_dt_dq_ND - (dddist_dt_dq(cls.rmodel, cls.cmodel, cls.x)).reshape(7,)
    #         ),
    #         0,
    #         places=2,
    #         msg=f"The derivative of the collision velocity w.r.t q is not equal to the one from numdiff. \n The value of the numdiff is : \n {cls.dddist_dt_dq_ND}\n and the value computed is : \n {dddist_dt_dq(cls.rmodel, cls.cmodel, cls.x).reshape(7,)}",
    #     )

    # def test_dh1_dq(cls):
        
        # cls.assertAlmostEqual(
    #         np.linalg.norm(
    #             cls.dddist_dt_dq_ND - dh1_dq(cls.rmodel, cls.cmodel, cls.q)
    #         ),
    #         0,
    #         places=2,
    #         msg=f"The derivative of the collision velocity w.r.t q is not equal to the one from numdiff. \n The value of the numdiff is : \n {cls.dddist_dt_dq_ND}\n and the value computed is : \n {dddist_dt_dq(cls.rmodel, cls.cmodel, cls.x).reshape(7,)}",
    #     )
        # print(np.linalg.norm(cls.dh1_dq_ND))
        # print(np.linalg.norm(cls.dh2_dq_ND))



def finite_diff_time(q, v, f, h=1e-8):
    return (f(q + h * v) - f(q)) / h


def numdiff(f, q, h=1e-4):
    j_diff = np.zeros(len(q))
    fx = f(q)
    for i in range(len(q)):
        e = np.zeros(len(q))
        e[i] = h
        j_diff[i] = (f(q + e) - fx) / e[i]
    return j_diff


def finite_difference_jacobian(f, x, h=1e-6):
    n_input = len(x)  # size of the input
    fx = f(x)  # evaluate function at x
    n_output = len(fx)  # size of the output

    jacobian = np.zeros((n_input, n_output))

    for i in range(n_input):
        x_forward = np.copy(x)
        x_backward = np.copy(x)

        x_forward[i] += h
        x_backward[i] -= h

        f_forward = f(x_forward).flatten()
        f_backward = f(x_backward).flatten()

        jacobian[i, :] = (f_forward - f_backward) / (2 * h)

    return jacobian


def numdiff_matrix(f, q, h=1e-6):
    fx = f(q).reshape(len(f(q)))
    j_diff = np.zeros((len(fx), len(q)))
    for i in range(len(q)):
        e = np.zeros(len(q))
        e[i] = h
        j_diff[i, :] = (f(q + e).reshape(len(fx)) - fx) / e[i]
    return j_diff


if __name__ == "__main__":
    unittest.main()
