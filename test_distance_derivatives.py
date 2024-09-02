import unittest
import numpy as np

import pinocchio as pin

from wrapper_panda import PandaWrapper
from distance_derivatives import dist, ddist_dq, ddist_dt, cp, dX_dq, dddist_dt_dq, h1, h2, A, dA_dt, R, dR_dt, c, dc_dt
from viewer import create_viewer, add_sphere_to_viewer

s = np.random.randint(1000)
pin.seed(s)
print(s)

class TestDistOpt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        def R1(rmodel, cmodel, q):
            return R(rmodel, cmodel, q, "obstacle")
        def R2(rmodel, cmodel, q):
            return R(rmodel, cmodel, q, "ellips_rob")
        def dR1_dt(rmodel, cmodel, x):
            return dR_dt(rmodel, cmodel, x, "obstacle")
        def dR2_dt(rmodel, cmodel, x):
            return dR_dt(rmodel, cmodel, x, "ellips_rob")
        def A1(rmodel, cmodel, q):
            return A(rmodel, cmodel, q, "obstacle")
        def A2(rmodel, cmodel, q):
            return A(rmodel, cmodel, q, "ellips_rob")
        def dA1_dt(rmodel, cmodel, x):
            return dA_dt(rmodel, cmodel, x, "obstacle")
        def dA2_dt(rmodel, cmodel, x):
            return dA_dt(rmodel, cmodel, x, "ellips_rob")

        # # OBS CONSTANTS
        # cls.PLACEMENT_OBS = pin.SE3(pin.utils.rotate("x", 0), np.array([0, 0, 2]))
        # cls.DIM_OBS = [0.15, 0.1, 0.2]

        # # ELLIPS ON THE ROBOT
        # cls.PLACEMENT_ROB = pin.SE3(pin.utils.rotate("x", 0), np.array([0, 0, 0]))
        # cls.DIM_ROB = [0.1, 0.2, 0.15]
        
        # OBS CONSTANTS
        cls.PLACEMENT_OBS = pin.SE3(pin.utils.rotate("x", np.pi/2), np.array([0, -0.0, 1.2]))
        cls.DIM_OBS = [0.1, 0.15, 0.1]

        # ELLIPS ON THE ROBOT
        cls.PLACEMENT_ROB = pin.SE3(np.eye(3), np.array([0, 0, 0]))
        cls.DIM_ROB =  [0.1, 0.1, 0.15]
        
        # Creating the robot
        robot_wrapper = PandaWrapper()
        cls.rmodel, cls.cmodel, _ = robot_wrapper()

        cls.cmodel = robot_wrapper.add_2_ellips(
            cls.cmodel,
            placement_obs=cls.PLACEMENT_OBS,
            dim_obs=cls.DIM_OBS,
            placement_rob=cls.PLACEMENT_ROB,
            dim_rob=cls.DIM_ROB,
        )
        cls.q = pin.randomConfiguration(cls.rmodel)
        cls.v = pin.randomConfiguration(cls.rmodel)
        cls.x = np.concatenate((cls.q, cls.v))

        # cls.x = np.array([-0.13618401593655072, 1.4548323140534223, -0.7455721959187626, 0.9802847672458945, 0.20975276500006032, 0.4790169063529318, -0.0010387435491745235, -6.909107593655072, 9.502458405342223, 7.048769408123746, 23.785128724589452, -21.444000499993965, 2.354105635293183, 0.35238764508254505])
        # cls.q = cls.x[:cls.rmodel.nq]
        # cls.v = cls.x[cls.rmodel.nq:]
        
        
        print(cls.q)
        cls.ddist_dt_ND = finite_diff_time(
            cls.q,
            cls.v,
            lambda variable: dist(
                cls.rmodel,
                cls.cmodel,
                variable,
            ),
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

        cls.cp = cp(cls.rmodel, cls.cmodel, cls.q)
        
        cls.c_dot_ND = finite_diff_time(
            cls.q,
            cls.v,
            lambda variable: c(cls.rmodel, cls.cmodel, variable),
        )

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

        viz = create_viewer(cls.rmodel, cls.cmodel, cls.cmodel)
        viz.display(cls.q)
        add_sphere_to_viewer(viz, "cp1", 1.5e-2, cls.cp[:3], color=1000)
        add_sphere_to_viewer(viz, "cp2", 1.5e-2, cls.cp[3:], color=100000)

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
                - dR_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), "obstacle")
            ),
            0,
            places=4,
            msg=f"The time derivative of the rotation matrix is not equal to the one from numdiff. \n The value of the numdiff is : \n {cls.R1_dot_ND}\n and the value computed is : \n {dR_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), "obstacle")}",
        )
        
    def test_dR2_dt(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.R2_dot_ND
                - dR_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), "ellips_rob")
            ),
            0,
            places=4,
            msg=f"The time derivative of the rotation matrix is not equal to the one from numdiff. \n The value of the numdiff is : \n {cls.R2_dot_ND}\n and the value computed is : \n {dR_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), "ellips_rob")}",
        )
        
    def test_dA1_dt(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.A1_dot_ND
                - dA_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), "obstacle")
            ),
            0,
            places=4,
            msg=f"The time derivative of the rotation matrices is not equal to the one from numdiff. \n The value of the numdiff is : \n {cls.A1_dot_ND}\n and the value computed is : \n {dA_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), "obstacle")}",
        )

    def test_dA2_dt(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.A2_dot_ND
                - dA_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), "ellips_rob")
            ),
            0,
            places=3,
            msg=f"The time derivative of the rotation matrices is not equal to the one from numdiff. \n The value of the numdiff is : \n {cls.A2_dot_ND}\n and the value computed is : \n {dA_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)), "ellips_rob")}",
        )
        

    def test_dc_dt(cls):
        cls.assertAlmostEqual(
            np.linalg.norm(
                cls.c_dot_ND 
                - dc_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)))
            ),
            0,
            places=4,
            msg=f"The derivative of the closest point w.r.t t is not equal to the one from numdiff. \n The value of the numdiff is : \n {cls.c_dot_ND}\n and the value computed is : \n {dc_dt(cls.rmodel, cls.cmodel, np.concatenate((cls.q, cls.v)))}",
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



def finite_diff_time(q, v, f, h=1e-9):
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
