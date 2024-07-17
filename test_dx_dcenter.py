import numpy as np
from qcqp_solver import EllipsoidOptimization
np.random.seed(0)
np.set_printoptions(3)



def numdiff(f, inX, h=1e-6):
    # Computes the Jacobian of a function returning a 1d array
    f0 = np.array(f(inX)).copy()
    x = inX.copy()
    df_dx = np.zeros((f0.size, len(x)))
    for ix in range(len(x)):
        x[ix] += h
        df_dx[:, ix] = (f(x) - f0) / h
        x[ix] = inX[ix]
    return df_dx

A = 3 * np.array([[1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
B = np.array([[0.1, 0, 0], [0, 0.6, 0], [0, 0, 1]])


def lagrangian(x, lambda_, center):
    x1 = x[:3]
    x2 = x[3:]
    lambda_1 = lambda_[0]
    lambda_2 = lambda_[1]
    center_1 = center[:3]
    center_2 = center[3:]

    return (0.5 * np.linalg.norm(x1 - x2, 2)**2 + lambda_1 * ((x1 - center_1).T @ A @ (x1 - center_1) - 1) / 2 + lambda_2 * ((x2 - center_2).T @ B @ (x2 - center_2)-1) / 2).item()



def h1(x, center):
    x1 = x[:3]
    center_1 = center[:3]
    return (((x1 - center_1).T @ A @ (x1 - center_1) - 1) / 2 ).item()

def h2(x, center):
    x2 = x[3:]
    center_2 = center[3:]
    return (((x2 - center_2).T @ B @ (x2 - center_2)-1) / 2).item()


def dh1_dx(x, center):
    x1 = x[:3]
    center_1 = center[:3]
    g1 = A @ (x1 - center_1)
    g2 = np.zeros(3)
    return np.concatenate((g1, g2))



def dh1_dcenter(x, center):
    x1 = x[:3]
    center_1 = center[:3]
    g1 = - A @ (x1 - center_1)
    g2 = np.zeros(3)
    return np.concatenate((g1, g2))


def dh2_dcenter(x, center):
    x2 = x[3:]
    center_2 = center[3:]
    g1 = np.zeros(3)
    g2 = - B @ (x2 - center_2)
    return np.concatenate((g1, g2))


def dh2_dx(x, center):
    x2 = x[3:]
    center_2 = center[3:]
    g1 = np.zeros(3)
    g2 = B @ (x2 - center_2)
    return np.concatenate((g1, g2))


def grad_x(x, lambda_, center):
    x1 = x[:3]
    x2 = x[3:]
    lambda_1 = lambda_[0]
    lambda_2 = lambda_[1]
    center_1 = center[:3]
    center_2 = center[3:]
    g1 = (x1 - x2) + lambda_1 * A @ (x1 - center_1)
    g2 = - (x1 - x2)  + lambda_2 * B @ (x2 - center_2) 
    return np.concatenate((g1, g2))

def hessian_xx(x, lambda_, center):
    lambda_1 = lambda_[0]
    lambda_2 = lambda_[1]

    H = np.zeros((6, 6))
    H[:3, 3:] =  - np.eye(3)
    H[3:, :3] =  - np.eye(3)
    H[:3, :3] =  np.eye(3) +  lambda_1 * A
    H[3:, 3:] =  np.eye(3) +  lambda_2 * B
    
    return H



def hessian_center_x(x, lambda_, center):
    lambda_1 = lambda_[0]
    lambda_2 = lambda_[1]

    H = np.zeros((6, 6))
    H[:3, :3] =  - lambda_1 * A
    H[3:, 3:] =  - lambda_2 * B
    
    return H

def func_lambda_annalytical(center):
    center_1 = center[:3]
    center_2 = center[3:]
    qcqp_solver = EllipsoidOptimization()
    qcqp_solver.setup_problem(center_1, A, center_2, B)
    qcqp_solver.solve_problem(warm_start_primal=center)
    x1, x2 = qcqp_solver.get_optimal_values()

    l1 = - np.linalg.norm(x1 - x2, 2) ** 2 /  ((x1- x2).T @ A @ (x1 - center_1)).item()
    l2 = np.linalg.norm(x1 - x2, 2) ** 2 /  ((x1- x2).T @ B @ (x2 - center_2)).item()
    return np.array([l1, l2])


def func_distance_annalytical(center):
    center_1 = center[:3]
    center_2 = center[3:]
    qcqp_solver = EllipsoidOptimization()
    qcqp_solver.setup_problem(center_1, A, center_2, B)
    qcqp_solver.solve_problem(warm_start_primal=center)
    x1, x2 = qcqp_solver.get_optimal_values()

    return 0.5 * np.linalg.norm(x1 - x2, 2) ** 2


def x_star(center):
    center_1 = center[:3]
    center_2 = center[3:]
    qcqp_solver = EllipsoidOptimization()
    qcqp_solver.setup_problem(center_1, A, center_2, B)
    qcqp_solver.solve_problem(warm_start_primal=center)

    xSol1, xSol2 = qcqp_solver.get_optimal_values()
    # distance = qcqp_solver.get_minimum_cost()
    # lambda1, lambda2 = qcqp_solver.get_dual_values()
    return np.concatenate((xSol1, xSol2))


def func_lambda(center):
    center_1 = center[:3]
    center_2 = center[3:]
    qcqp_solver = EllipsoidOptimization()
    qcqp_solver.setup_problem(center_1, A, center_2, B)
    qcqp_solver.solve_problem(warm_start_primal=center)

    lambda1, lambda2 = qcqp_solver.get_dual_values()

    return np.array([lambda1, lambda2])

def func_distance(center):
    center_1 = center[:3]
    center_2 = center[3:]
    qcqp_solver = EllipsoidOptimization()
    qcqp_solver.setup_problem(center_1, A, center_2, B)
    qcqp_solver.solve_problem(warm_start_primal=center)
    return qcqp_solver.get_minimum_cost()



def dx_dcenter(center):
    center_1 = center[:3]
    center_2 = center[3:]
    qcqp_solver = EllipsoidOptimization()
    qcqp_solver.setup_problem(center_1, A, center_2, B)
    qcqp_solver.solve_problem(warm_start_primal=center)
    xSol1, xSol2 = qcqp_solver.get_optimal_values()
    lambda1, lambda2 = qcqp_solver.get_dual_values()

    x = np.concatenate((xSol1, xSol2))
    lambda_ = np.array([lambda1, lambda2])

    M_matrix = np.zeros((8, 8))
    N_matrix = np.zeros((8, 6))

    dh1_dx_ = dh1_dx(x, center)
    dh2_dx_ = dh2_dx(x, center)

    M_matrix[:6, :6] = hessian_xx(x, lambda_, center)
    M_matrix[:6, 6] = dh1_dx_
    M_matrix[:6, 7] = dh2_dx_
    M_matrix[6, :6] = dh1_dx_.T
    M_matrix[7, :6] = dh2_dx_.T
  
    dh1_do_ = dh1_dcenter(x, center)
    dh2_do_ = dh2_dcenter(x, center)

    N_matrix[:6, :] = hessian_center_x(x, lambda_, center)
    N_matrix[6, :] = dh1_do_
    N_matrix[7, :] = dh2_do_
    
    dy = - np.linalg.solve(M_matrix, N_matrix)

    return dy[:6]





x = np.random.random(6)
lambda_ = np.random.random(2)
center = np.random.random(6)
center[3:] += 10


grad_x_ND = numdiff(lambda variable:lagrangian(variable, lambda_, center), x)
hessian_center_x_ND = numdiff(lambda variable:grad_x(x, lambda_, variable), center)
hessian_xx_ND = numdiff(lambda variable:grad_x(variable, lambda_, center), x)
dh1_dx_ND = numdiff(lambda variable:h1(variable, center), x)
dh2_dx_ND = numdiff(lambda variable:h2(variable, center), x)
dh1_dcenter_ND = numdiff(lambda variable:h1(x, variable), center)
dh2_dcenter_ND = numdiff(lambda variable:h2(x, variable), center)

dx_dcenter_ND = numdiff(lambda variable:x_star(variable), center)


set_tol = 1e-4


assert np.linalg.norm(grad_x_ND - grad_x(x, lambda_, center)) < set_tol
assert np.linalg.norm(hessian_center_x_ND - hessian_center_x(x, lambda_, center)) < set_tol
assert np.linalg.norm(hessian_xx_ND - hessian_xx(x, lambda_, center)) < set_tol
assert np.linalg.norm(dh1_dx_ND - dh1_dx(x, center)) < set_tol
assert np.linalg.norm(dh2_dx_ND - dh2_dx(x, center)) < set_tol
assert np.linalg.norm(dh1_dcenter_ND - dh1_dcenter(x, center)) < set_tol
assert np.linalg.norm(dh2_dcenter_ND - dh2_dcenter(x, center)) < set_tol
assert  np.linalg.norm(func_lambda_annalytical(center)- func_lambda(center)) < set_tol
assert  np.linalg.norm(func_distance_annalytical(center)- func_distance(center)) < set_tol
assert  np.linalg.norm(dx_dcenter_ND - dx_dcenter(center) ) < set_tol