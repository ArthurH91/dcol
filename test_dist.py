# ellipsoid_optimization.py
import numpy as np
import casadi as ca
import hppfcl
import pinocchio as pin

np.random.seed(0)

# Define initial positions for the centers of the two ellipsoids
x0_1 = np.random.randn(3)
print(x0_1)
x0_2 = 10 * np.random.randn(3)
print(x0_2)
# Define the radii for the ellipsoids
radiiA = np.random.randn(3)
radiiB = np.random.randn(3)

# Construct matrices A and B for ellipsoid constraints
# A = np.diag([1 / r**2 for r in radiiA])
# B = np.diag([1 / r**2 for r in radiiB])
A = np.diag([1]*3)
B = np.diag([1]*3)

# Generate random rotation matrices using Pinocchio
R_A = pin.SE3.Random().rotation
R_B = pin.SE3.Random().rotation

R_A = np.eye(3)
R_B = np.eye(3)
# Calculate rotated matrices
A_rot = R_A.T @ A @ R_A
B_rot = R_B.T @ B @ R_B

# Setup optimization problem with CasADi
opti = ca.Opti()
x1 = opti.variable(3)
x2 = opti.variable(3)

# Define the cost function (distance between points)
totalcost = ca.norm_2(x1 - x2)

# Define constraints for the ellipsoids
con1 = (x1 - x0_1).T @ A_rot @ (x1 - x0_1) <= 1
con2 = (x2 - x0_2).T @ B_rot @ (x2 - x0_2) <= 1
opti.subject_to([con1, con2])

opti.solver('ipopt')
opti.minimize(totalcost)

# Apply warm start values
opti.set_initial(x1, x0_1)
opti.set_initial(x2, x0_2)

# Solve the optimization problem
try:
    solution = opti.solve()
    x1_sol = opti.value(x1)
    x2_sol = opti.value(x2)
except RuntimeError as e:
    print(f"Solver failed: {e}")
    x1_sol = opti.debug.value(x1)
    x2_sol = opti.debug.value(x2)
    print("Debug values:")
    print("x1:", x1_sol)
    print("x2:", x2_sol)
    print("Total cost:", opti.debug.value(totalcost))
    raise

# Setup ellipsoids and SE3 transformations in HPP-FCL
ellipsA = hppfcl.Ellipsoid(*radiiA)
ellipsB = hppfcl.Ellipsoid(*radiiB)
centerA = pin.SE3(rotation=R_A, translation=x0_1)
centerB = pin.SE3(rotation=R_B, translation=x0_2)

# Compute distances and nearest points using HPP-FCL
req = hppfcl.DistanceRequest()
res = hppfcl.DistanceResult()
dist = hppfcl.distance(ellipsA, centerA, ellipsB, centerB, req, res)

# Print results for comparison
print(f"x1 CasADi: {x1_sol} || x1 HPP-FCL: {res.getNearestPoint1()} || x1 DIFF: {x1_sol - res.getNearestPoint1()}")
print(f"x2 CasADi: {x2_sol} || x2 HPP-FCL: {res.getNearestPoint2()} || x2 DIFF: {x2_sol - res.getNearestPoint2()}")
print(f"Distance CasAdi: {opti.debug.value(totalcost)} || Distance HPP-FCL {dist} || DIFF {opti.debug.value(totalcost) - dist}")


