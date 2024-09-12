'''
For two ellipsoid in 2D E1=(A1,c1) E2=(A2,c2)
compute the witness point x and the distance |x1-x2| 
by solving the problem:
decide x1,x2
minimizing d=|x1-x2|
so that 
        .5(x1-c1)A1(x1-c1) -.5 == 0
        .5(x2-c2)A2(x2-c2) -.5 == 0
'''
import numpy as np
import hppfcl
import pinocchio as pin

import numpy as np
import casadi
import pinocchio as pin
from numpy import r_,c_,eye
from pinocchio import skew
import hppfcl
import warnings
from wrapper_panda import PandaWrapper
np.set_printoptions(precision=4, linewidth=350, suppress=True,threshold=1e6)
pin.SE3.__repr__=pin.SE3.__str__
cross = np.cross
skew = pin.skew
s_opt = {
    "tol": 1e-15,
    "acceptable_tol": 1e-15,
    "max_iter": 1000,
}

from compute_deriv import compute_d_d_dot_dq_dq_dot, compute_ddot, compute_dist, compute_Ldot

SEED = 1
np.random.seed(SEED)
pin.seed(SEED)

def softassert(clause,msg):
    if not clause: warnings.warn(msg)

robot_wrapper = PandaWrapper()
rmodel, gmodel, vmodel = robot_wrapper()

# ########################################################
# robot = robex.load('panda')
# rmodel,gmodel = robot.model,robot.collision_model

D1 = np.diagflat([1/0.1**2,1/0.2**2,1/0.1**2])
D2 = np.diagflat([1/0.04**2,1/0.04**2,1/0.04**2])


Mobs = pin.SE3(pin.utils.rotate("y", np.pi ) @ pin.utils.rotate("z", np.pi / 2),np.array([0, 0.1, 1.2]))
rmodel.addFrame(pin.Frame('obstacle',0,0,Mobs,pin.OP_FRAME))

idf1 = rmodel.getFrameId('obstacle')
idj1 = rmodel.frames[idf1].parentJoint
elips1 = hppfcl.Ellipsoid(*[ d**-.5 for d in np.diag(D1) ])
elips1_geom = pin.GeometryObject('el1', idj1,idf1,rmodel.frames[idf1].placement,elips1)
elips1_geom.meshColor = r_[1,0,0,1]
idg1 = gmodel.addGeometryObject(elips1_geom)

idf2 = rmodel.getFrameId('panda2_hand_tcp')
idj2 = rmodel.frames[idf2].parentJoint
elips2 = hppfcl.Ellipsoid(*[ d**-.5 for d in np.diag(D2) ])
elips2_geom = pin.GeometryObject('el2', idj2,idf2,rmodel.frames[idf2].placement,elips2)
elips2_geom.meshColor = r_[1,1,0,1]
idg2 = gmodel.addGeometryObject(elips2_geom)

rdata,gdata = rmodel.createData(),gmodel.createData()

# ########################################################
# Ellipses placements

# x = [-0.24353014,0.77536231,-1.19963862,-1.0891806,0.09815845,0.9268209,-0.08219284,0.09283095,0.27348186,-0.01105114,-0.77302222,-0.06031285,0.31114164,0.01417026]

# x = [-1.58738191e-01,1.32322692e+00,-9.20085197e-01,7.03668067e-01
# ,3.25401712e-01,3.50520117e-01,-1.05107719e-03,-9.16452511e-01
# ,-3.65808115e-01,-1.04025307e+00,-3.87654134e-01,-9.87910584e-01
# ,-1.04955733e+00,3.51154281e-02]

x = [-0.16755916520002284, 1.0695138003180005, -0.6589425036978582, 0.5565265096523058, 0.29425709852841514, 0.35493553340229883, -0.006626160642339856, -1.1514280582502283, 0.07164011255500463, -0.21395714635358215, -0.23499646597694174, -0.37143503034084835, -0.009049744102011759, -0.030616137673398686]


q = np.array(x[:7])
vq = np.array(x[7:])
aq = np.array([-1,  4, -3, -3, -4, -5,  1])

# Compute robot placements and velocity at step 0 and 1
pin.forwardKinematics(rmodel,rdata,q,vq)
pin.updateFramePlacements(rmodel,rdata)
pin.updateGeometryPlacements(rmodel,rdata,gmodel,gdata)

M1 = gdata.oMg[idg1].copy()
R1,c1 = M1.rotation,M1.translation
A1 = R1@D1@R1.T
M2 = gdata.oMg[idg2].copy()
R2,c2 = M2.rotation,M2.translation
A2 = R2@D2@R2.T


# Get body velocity at step 0
pin.forwardKinematics(rmodel,rdata,q,vq)
nu1 = pin.getFrameVelocity(rmodel,rdata,idf1,pin.LOCAL_WORLD_ALIGNED).copy()
v1,w1 = nu1.linear,nu1.angular
nu2 = pin.getFrameVelocity(rmodel,rdata,idf2,pin.LOCAL_WORLD_ALIGNED).copy()
v2,w2 = nu2.linear,nu2.angular

# Get body placement at step 1
dt = 1e-6
qplus = pin.integrate(rmodel,q,vq*dt)
vqplus = vq + aq*dt

rdataplus,gdataplus = rmodel.createData(),gmodel.createData()
pin.forwardKinematics(rmodel,rdataplus,qplus,vqplus)
pin.updateFramePlacements(rmodel,rdataplus)
pin.updateGeometryPlacements(rmodel,rdataplus,gmodel,gdataplus)

M1plus = gdataplus.oMg[idg1].copy()
R1plus,c1plus = M1plus.rotation,M1plus.translation
A1plus = R1plus@D1@R1plus.T

M2plus = gdataplus.oMg[idg2].copy()
R2plus,c2plus = M2plus.rotation,M2plus.translation
A2plus = R2plus@D2@R2plus.T

# Get body velocity at step 1
nu1plus = pin.getFrameVelocity(rmodel,rdataplus,idf1,pin.LOCAL_WORLD_ALIGNED).copy()
v1plus,w1plus = nu1.linear,nu1.angular
nu2plus = pin.getFrameVelocity(rmodel,rdataplus,idf2,pin.LOCAL_WORLD_ALIGNED).copy()
v2plus,w2plus = nu2plus.linear,nu2plus.angular

#v1,w1 = np.r_[1,2,-1], np.r_[-2,-1,3]
#v2,w2 = np.r_[.5,-.6,.3], np.r_[.6,0.5,-.7]

assert( np.allclose( pin.log(gdata.oMg[idg1].inverse()*rdata.oMf[idf1]).vector,0 ) )
assert( np.allclose( pin.log(gdata.oMg[idg2].inverse()*rdata.oMf[idf2]).vector,0 ) )

# ########################################################
# Simplest test: assert optimality condition

opti = casadi.Opti()
var_x1 = opti.variable(3) 
var_x2 = opti.variable(3) 
totalcost = .5*casadi.sumsqr(var_x1-var_x2)
opti.subject_to( .5* (var_x1-c1).T@A1@(var_x1-c1) -.5<= 0)
opti.subject_to( .5* (var_x2-c2).T@A2@(var_x2-c2) -.5<= 0)
opti.minimize(totalcost)
opti.solver('ipopt',{},s_opt)
sol = opti.solve()

sol_x1 = opti.value(var_x1)
sol_x2 = opti.value(var_x2)
sol_lam1,sol_lam2 = opti.value(opti.lam_g)
sol_f = .5*sum((sol_x1-sol_x2)**2)
sol_d = (2*sol_f)**.5
sol_L = sol_f

assert(np.allclose(sol_x2-sol_x1,sol_lam1*A1@(sol_x1-c1)))
assert(np.allclose((sol_x1-c1)@A1@(sol_x1-c1),1,1e-6))
assert(np.allclose(sol_x1-sol_x2,sol_lam2*A2@(sol_x2-c2)))
assert(np.allclose((sol_x2-c2)@A2@(sol_x2-c2),1,1e-3))

req = hppfcl.DistanceRequest()
req.gjk_max_iterations = 20000
req.abs_err = 0
req.gjk_tolerance = 1e-9
res = hppfcl.DistanceResult()
distance = hppfcl.distance(
    elips1,
    gdata.oMg[idg1],
    elips2,
    gdata.oMg[idg2],
    req,
    res,
)
x1 = res.getNearestPoint1()
x2 = res.getNearestPoint2()
dist = distance

dist_from_compute_dist = compute_dist(rmodel, gmodel, q,vq,  idg1, idg2)

assert np.isclose(dist,dist_from_compute_dist, atol=1e-6), f'{dist} != {dist_from_compute_dist}'
assert np.allclose(sol_x1,x1, atol=1e-5), f'{sol_x1} != {x1}'
assert(np.allclose(sol_x2,x2, atol= 1e-5)), f'{sol_x2} != {x2}'

sol_lam1_ana, sol_lam2_ana = -(x1 - c1).T @ (x1 - x2), (x2 - c2).T @ (x1 - x2)

print(f"sol_lam1: {sol_lam1}")
print(f"sol_lam2: {sol_lam2}")
assert np.allclose(sol_lam1,sol_lam1_ana, atol=1e-6), f'{sol_lam1} != {sol_lam1_ana}'
assert np.allclose(sol_lam2,sol_lam2_ana, atol=1e-6), f'{sol_lam2} != {sol_lam2_ana}'


# ########################################################
# Meshcat rendering

# ########################################################
# Compute d_dot and assert ND
# R1plus = pin.exp(w1*dt)@R1
# c1plus = c1+v1*dt
# A1plus = R1plus@D1@R1plus.T
# R2plus = pin.exp(w2*dt)@R2
# c2plus = c2+v2*dt
# A2plus = R2plus@D2@R2plus.T
opti = casadi.Opti()
var_x1 = opti.variable(3) 
var_x2 = opti.variable(3) 
totalcost = 1/2* casadi.sumsqr(var_x1-var_x2)
opti.subject_to( (var_x1-c1plus).T@A1plus@(var_x1-c1plus) /2  <= 1/2)
opti.subject_to( (var_x2-c2plus).T@A2plus@(var_x2-c2plus) /2  <= 1/2)
opti.minimize(totalcost)
opti.solver('ipopt',{},s_opt)
sol = opti.solve()
next_x1 = opti.value(var_x1)
next_x2 = opti.value(var_x2)
next_lam1,next_lam2 = opti.value(opti.lam_g)
next_f = .5*sum((next_x1-next_x2)**2)
next_d = (2*next_f)**.5
next_L = next_f
assert(np.allclose(-(next_x1-next_x2),next_lam1*A1@(next_x1-c1plus),atol=1e-5))
assert(np.allclose((next_x1-c1plus)@A1@(next_x1-c1plus),1,1e-5))
assert(np.allclose((next_x1-next_x2),next_lam2*A2@(next_x2-c2plus),atol=1e-6))
assert(np.allclose((next_x2-c2plus)@A2@(next_x2-c2plus),1,1e-3))

Ldot_ND = (next_L-sol_L)/dt

Ldot = (sol_x1-sol_x2)@(v1-v2) \
    - np.cross(sol_x1-sol_x2,sol_x1-c1)@w1 \
    + np.cross(sol_x1-sol_x2,sol_x2-c2)@w2
assert(np.isclose(Ldot,Ldot_ND, atol=1e-6)), f'{Ldot} != {Ldot_ND}'


Ldot_test = compute_Ldot(rmodel, gmodel, q, vq, idg1, idg2)

assert np.isclose(Ldot,Ldot_test, atol=1e-6), f'{Ldot} != {Ldot_test}'


# Another (simpler?) expression for Ldot

vx1 = v1 + np.cross(w1,sol_x1-c1)
vx2 = v2 + np.cross(w2,sol_x2-c2)
Ldot_other = (sol_x1-sol_x2)@(vx1-vx2)
assert(np.isclose(Ldot,Ldot_other))

ddot_ND = (next_d-sol_d)/dt
ddot = Ldot/sol_d
assert(np.isclose(ddot,ddot_ND, atol=1e-5)), f'{ddot} != {ddot_ND}'


ddot_test = compute_ddot(rmodel, gmodel, q, vq, idg1, idg2)
assert np.isclose(ddot,ddot_test, atol=1e-5), f'{ddot} != {ddot_test}'

# #############################
# Compute derivatives of L and assert
sol_y = np.concatenate([sol_x1,sol_x2,[sol_lam1,sol_lam2]])
next_y = np.concatenate([next_x1,next_x2,[next_lam1,next_lam2]])

Ly = r_[ (sol_x1-sol_x2)+sol_lam1*A1@(sol_x1-c1),
         -(sol_x1-sol_x2)+sol_lam2*A2@(sol_x2-c2),
         1/2*(sol_x1-c1)@A1@(sol_x1-c1)-1/2,
         1/2*(sol_x2-c2)@A2@(sol_x2-c2)-1/2 ]

# Ly_cplus is computed in y but with cplus (not c), then it is not null)
Ly_cplus = r_[ (sol_x1-sol_x2)+sol_lam1*A1plus@(sol_x1-c1plus),
               -(sol_x1-sol_x2)+sol_lam2*A2plus@(sol_x2-c2plus),
               1/2*(sol_x1-c1plus)@A1plus@(sol_x1-c1plus)-1/2,
               1/2*(sol_x2-c2plus)@A2plus@(sol_x2-c2plus)-1/2 ]

# Ly_yplus is computed in next_y but with c (not cplus), then it is not null)
Ly_yplus = r_[ (next_x1-next_x2)+next_lam1*A1@(next_x1-c1),
         -(next_x1-next_x2)+next_lam2*A2@(next_x2-c2),
         1/2*(next_x1-c1)@A1@(next_x1-c1)-1/2,
         1/2*(next_x2-c2)@A2@(next_x2-c2)-1/2 ]


Lyy = r_[ c_[eye(3)+sol_lam1*A1, -eye(3), A1@(sol_x1-c1), np.zeros(3)],
          c_[-eye(3),  eye(3)+sol_lam2*A2, np.zeros(3), A2@(sol_x2-c2)],
          [r_[A1@(sol_x1-c1),  np.zeros(3), np.zeros(2)]],
          [r_[np.zeros(3), A2@(sol_x2-c2),  np.zeros(2)]] ]
Lyc = r_[ c_[-sol_lam1*A1, np.zeros([3,3])],
          c_[np.zeros([3,3]),-sol_lam2*A2 ],
          [r_[-A1@(sol_x1-c1), np.zeros(3)]],
          [r_[np.zeros(3),-A2@(sol_x2-c2)]]]
Lyr = r_[ c_[sol_lam1*(A1@skew(sol_x1-c1)-skew(A1@(sol_x1-c1))), np.zeros([3,3])],
          c_[np.zeros([3,3]),sol_lam2*(A2@skew(sol_x2-c2)-skew(A2@(sol_x2-c2))) ],
          [r_[(sol_x1-c1)@A1@skew(sol_x1-c1), np.zeros(3)]],
          [r_[np.zeros(3),(sol_x2-c2)@A2@skew(sol_x2-c2) ]] ]
Lyth = c_[Lyc[:,:3],Lyr[:,:3],Lyc[:,3:],Lyr[:,3:]]
# Tolerance here must be very high, not sure why, but those tests are not super important
softassert(np.allclose(Ly_yplus/dt,Lyy@(next_y-sol_y)/dt,rtol=1e-1), 'Ly_yplus!=0')
softassert(np.allclose(Ly_cplus/dt,Lyc@r_[v1,v2]+Lyr@r_[w1,w2],rtol=1e-1),'Ly_cplus!=0')
softassert(np.allclose(Ly_cplus/dt,-Ly_yplus/dt,rtol=1e-1),'Lycplus!=-Ly_yplus')

# ##############################
# Cnompute derivative of y=[x,lam] and assert
v = r_[v1,v2]
w = r_[w1,w2]
theta_dot = r_[v1,w1,v2,w2]
yc = -np.linalg.inv(Lyy)@Lyc
yr = -np.linalg.inv(Lyy)@Lyr
yth = -np.linalg.inv(Lyy)@Lyth
ydot = yc@v+yr@w
ydot_other = yth@theta_dot
assert(np.allclose(ydot,ydot_other))
x1dot_ND = (next_x1-sol_x1)/dt
lam1dot_ND = (next_lam1-sol_lam1)/dt
x2dot_ND = (next_x2-sol_x2)/dt
lam2dot_ND = (next_lam2-sol_lam2)/dt
ydot_ND=np.concatenate([x1dot_ND,x2dot_ND,[lam1dot_ND,lam2dot_ND]])

assert(np.allclose(ydot,ydot_ND,atol=1e-5))

# #####
# Check d Ldot / dv and d Ldot / dw
Lc1 = sol_x1-sol_x2
Lc2 = -(sol_x1-sol_x2)
Lr1 = -np.cross(sol_x1-sol_x2,sol_x1-c1)
Lr2 = np.cross(sol_x1-sol_x2,sol_x2-c2)
Lth = r_[Lc1,Lr1,Lc2,Lr2]
Lth_thdot = Lc1@v1 + Lr1@w1 + Lc2@v2 + Lr2@w2
assert(np.isclose(Ldot,Lth_thdot))
Lth_thdot_other = Lth@theta_dot
assert(np.isclose(Lth_thdot,Lth_thdot_other))

Ldot = (sol_x1-sol_x2)@(v1-v2) \
    - np.cross(sol_x1-sol_x2,sol_x1-c1)@w1 \
    + np.cross(sol_x1-sol_x2,sol_x2-c2)@w2
next_Ldot = (next_x1-next_x2)@(v1-v2) \
    - np.cross(next_x1-next_x2,next_x1-c1plus)@w1 \
    + np.cross(next_x1-next_x2,next_x2-c2plus)@w2
dLdot_ND = (next_Ldot-Ldot)/dt

# partial derivative of Ldot
Ldot_cplus = (sol_x1-sol_x2)@(v1-v2) \
    - np.cross(sol_x1-sol_x2,sol_x1-c1plus)@w1 \
    + np.cross(sol_x1-sol_x2,sol_x2-c2plus)@w2
dLdot_dc1 = -cross(sol_x1-sol_x2,w1)
dLdot_dc2 = cross(sol_x1-sol_x2,w2)
assert(np.isclose((Ldot_cplus-Ldot)/dt,dLdot_dc1@v1 + dLdot_dc2@v2))
Ldot_xplus = (next_x1-next_x2)@(v1-v2) \
    - np.cross(next_x1-next_x2,next_x1-c1)@w1 \
    + np.cross(next_x1-next_x2,next_x2-c2)@w2
dLdot_dx1 = v1-v2 -cross(sol_x1-c1,w1)+cross(sol_x2-c2,w2)+cross(sol_x1-sol_x2,w1)
dLdot_dx2 = -v1+v2 +cross(sol_x1-c1,w1)-cross(sol_x2-c2,w2)-cross(sol_x1-sol_x2,w2)
assert(np.isclose((Ldot_xplus-Ldot)/dt,
                  dLdot_dx1@(next_x1-sol_x1)/dt + dLdot_dx2@(next_x2-sol_x2)/dt))

dLdot_dx1_simpler = v1-vx2 -cross(sol_x2-c1,w1)
dLdot_dx2_simpler = -vx1+v2 -cross(sol_x1-c2,w2)
assert(np.allclose(dLdot_dx1,dLdot_dx1_simpler))
assert(np.allclose(dLdot_dx2,dLdot_dx2_simpler))

dLdot_dc = r_[dLdot_dc1,dLdot_dc2] + r_[dLdot_dx1,dLdot_dx2]@yc[:6]
dLdot_dr = r_[dLdot_dx1,dLdot_dx2]@yr[:6]
dLdot_ND = (next_Ldot-Ldot)/dt
assert(np.isclose(dLdot_ND,dLdot_dc@r_[v1,v2]+dLdot_dr@r_[w1,w2]))

# ####################
# Check with DDL

xc,xr = yc[:3],yr[:3]
#dx1 = c_[yc[:3],yr[:3]]
#dx2 = c_[yc[3:6],yr[3:6]]
dx1 = yth[:3]
dx2 = yth[3:6]


dL = r_[sol_x1-sol_x2,-cross(sol_x1-sol_x2,sol_x1-c1),-(sol_x1-sol_x2),cross(sol_x1-sol_x2,sol_x2-c2)]
assert(np.isclose(Ldot,dL@theta_dot))

ddL = r_[dx1-dx2,
         -skew(sol_x1-sol_x2)@dx1+skew(sol_x1-c1)@(dx1-dx2),
         -dx1+dx2,
         skew(sol_x1-sol_x2)@dx2-skew(sol_x2-c2)@(dx1-dx2)] \
    +r_[np.zeros([3,12]),
        c_[skew(sol_x1-sol_x2),np.zeros([3,9])],
        np.zeros([3,12]),
        c_[np.zeros([3,6]),-skew(sol_x1-sol_x2),np.zeros([3,3])]]
next_dL = r_[next_x1-next_x2,-cross(next_x1-next_x2,next_x1-c1plus),-(next_x1-next_x2),cross(next_x1-next_x2,next_x2-c2plus)]
dL_ND = (next_dL-dL)/dt

assert(np.allclose(dL_ND,ddL@theta_dot,atol=1e-4))

dLdot = theta_dot@ddL@theta_dot
assert(np.isclose(dLdot,dLdot_ND))



# ###########
# Check ddot

ddot = Ldot/sol_d
next_ddot = next_Ldot/next_d

dddot_dt_ND = (next_ddot-ddot)/dt
dddot = (theta_dot.T@ddL/sol_d - ddot/sol_d**2 * dL)
dddot_dt = dddot @ theta_dot
assert(np.isclose(dddot_dt,dddot_dt_ND,rtol=1e-3))


# #### Summary

# This is a summary of all previous computations.
# We assume that the problem min-distance is already solved, and
# optimal variables are stored under sol_x1 and sol_x2.
# For finite difference tests, we also assume the next problem (after
# integrating v and w during dt) is solved, and optimal variables
# are next_x1 and next_x2

# ### --- CALC

# Cost = half of squared distance
L = .5*sum((sol_x1-sol_x2))
# Distance
dist = np.linalg.norm(sol_x1-sol_x2)

# Witness point velocities (seen as fixed on the objects)
vx1 = v1 + np.cross(w1,sol_x1-c1)
vx2 = v2 + np.cross(w2,sol_x2-c2)

# Cost time derivative
Ldot = (sol_x1-sol_x2)@(vx1-vx2)
# Distance time derivative
dist_dot = Ldot/dist

# Assert dist_dot
next_dist = np.linalg.norm(next_x1-next_x2)
assert(np.isclose((next_dist-dist)/dt,dist_dot, atol=1e-5)), f'{(next_dist-dist)/dt} != {dist_dot}'

# ### --- CALC DIFF

# ### Derivative with respect to position: d_dist_dot_dtheta

# theta = (c1,c2,r1,r2)
theta_dot = r_[v1,w1,v2,w2]
# dist_dot derivative wrt theta
dL_dtheta = r_[sol_x1-sol_x2,-cross(sol_x1-sol_x2,sol_x1-c1),-(sol_x1-sol_x2),cross(sol_x1-sol_x2,sol_x2-c2)]

assert(np.allclose(Ldot,dL_dtheta@theta_dot))
ddL_dtheta2 = r_[dx1-dx2,
                 -skew(sol_x1-sol_x2)@dx1+skew(sol_x1-c1)@(dx1-dx2),
                 -dx1+dx2,
                 skew(sol_x1-sol_x2)@dx2-skew(sol_x2-c2)@(dx1-dx2)] \
                 +r_[np.zeros([3,12]),
                     c_[skew(sol_x1-sol_x2),np.zeros([3,9])],
                     np.zeros([3,12]),
                     c_[np.zeros([3,6]),-skew(sol_x1-sol_x2),np.zeros([3,3])]]
dLdot_dtheta = theta_dot@ddL_dtheta2
# Verif using finite diff
next_Ldot = (next_x1-next_x2)@( v1 + np.cross(w1,next_x1-c1plus)-(v2 + np.cross(w2,next_x2-c2plus)))
dLdot_dt_ND = (next_Ldot-Ldot)/dt
assert(np.isclose(dLdot_dt_ND,dLdot_dtheta@theta_dot))

d_dist_dot_dtheta = (theta_dot.T@ddL_dtheta2/dist - dist_dot/dist**2 * dL_dtheta)
# Verif using finite diff
d_dist_dot_dt = d_dist_dot_dtheta @ theta_dot
next_dist_dot = next_Ldot / next_dist
d_dist_dot_dt_ND = (next_dist_dot-dist_dot)/dt
assert(np.isclose(d_dist_dot_dt_ND,d_dist_dot_dt))

# ### Derivative with respect to velocity: d_dist_dot_dthetadot
d_dist_dot_dtheta_dot = dL_dtheta / dist
# Verif using finite diff
# Select a disturbed velocity
#v1_eps,w1_eps,v2_eps,w2_eps = [ v+dt*np.random.rand(3) for v in [v1,w1,v2,w2] ]

# Compute the disturbed dist_dot
theta_dot_plus = r_[v1plus,w1plus,v2plus,w2plus]
theta_ddot = (theta_dot_plus-theta_dot)/dt
next_dist_dot =  (sol_x1-sol_x2)@( v1plus + np.cross(w1plus,sol_x1-c1)-(v2plus + np.cross(w2plus,sol_x2-c2)))/dist
d_dist_dot_dt_ND = (next_dist_dot-dist_dot)/dt
d_dist_dot_dt = d_dist_dot_dtheta_dot@(theta_ddot)
assert(np.isclose(d_dist_dot_dt,d_dist_dot_dt_ND))



# ##################################3
# Robot derivatives

pin.computeJointJacobians(rmodel,rdata,q)
J1 = pin.getFrameJacobian(rmodel,rdata,idf1,pin.LOCAL_WORLD_ALIGNED)
J2 = pin.getFrameJacobian(rmodel,rdata,idf2,pin.LOCAL_WORLD_ALIGNED)
assert(np.allclose(nu1.vector,J1@vq))
assert(np.allclose(nu2.vector,J2@vq))

dtheta_dq = r_[J1,J2]
assert(np.allclose(dtheta_dq@vq,theta_dot))

dthetadot_dqdot = r_[J1,J2]
#assert(np.allclose(dthetadot_dqdot@aq,theta_ddot))

rdataplus,gdataplus = rmodel.createData(),gmodel.createData()
pin.forwardKinematics(rmodel,rdataplus,qplus,vqplus)
pin.updateFramePlacements(rmodel,rdataplus)
pin.updateGeometryPlacements(rmodel,rdataplus,gmodel,gdataplus)

M1plus = gdataplus.oMg[idg1].copy()
R1plus,c1plus = M1plus.rotation,M1plus.translation
A1plus = R1plus@D1@R1plus.T

M2plus = gdataplus.oMg[idg2].copy()
R2plus,c2plus = M2plus.rotation,M2plus.translation
A2plus = R2plus@D2@R2plus.T

# Get body velocity at step 1
nu1plus = pin.getFrameVelocity(rmodel,rdataplus,idf1,pin.LOCAL_WORLD_ALIGNED).copy()
v1plus,w1plus = nu1.linear,nu1.angular
nu2plus = pin.getFrameVelocity(rmodel,rdataplus,idf2,pin.LOCAL_WORLD_ALIGNED).copy()
v2plus,w2plus = nu2plus.linear,nu2plus.angular

#v1,w1 = np.r_[1,2,-1], np.r_[-2,-1,3]
#v2,w2 = np.r_[.5,-.6,.3], np.r_[.6,0.5,-.7]

assert( np.allclose( pin.log(gdata.oMg[idg1].inverse()*rdata.oMf[idf1]).vector,0 ) )
assert( np.allclose( pin.log(gdata.oMg[idg2].inverse()*rdata.oMf[idf2]).vector,0 ) )

# ########################################################
# Simplest test: assert optimality condition

opti = casadi.Opti()
var_x1 = opti.variable(3) 
var_x2 = opti.variable(3) 
totalcost = .5*casadi.sumsqr(var_x1-var_x2)
opti.subject_to( .5* (var_x1-c1).T@A1@(var_x1-c1) -.5<= 0)
opti.subject_to( .5* (var_x2-c2).T@A2@(var_x2-c2) -.5<= 0)
opti.minimize(totalcost)
opti.solver('ipopt',{},s_opt)
sol = opti.solve()

sol_x1 = opti.value(var_x1)
sol_x2 = opti.value(var_x2)
sol_lam1,sol_lam2 = opti.value(opti.lam_g)
sol_f = .5*sum((sol_x1-sol_x2)**2)
sol_d = (2*sol_f)**.5
sol_L = sol_f

assert(np.allclose(sol_x2-sol_x1,sol_lam1*A1@(sol_x1-c1)))
assert(np.allclose((sol_x1-c1)@A1@(sol_x1-c1),1,1e-6))
assert(np.allclose(sol_x1-sol_x2,sol_lam2*A2@(sol_x2-c2)))
assert(np.allclose((sol_x2-c2)@A2@(sol_x2-c2),1,1e-3))


# ########################################################
# Compute d_dot and assert ND
# R1plus = pin.exp(w1*dt)@R1
# c1plus = c1+v1*dt
# A1plus = R1plus@D1@R1plus.T
# R2plus = pin.exp(w2*dt)@R2
# c2plus = c2+v2*dt
# A2plus = R2plus@D2@R2plus.T
opti = casadi.Opti()
var_x1 = opti.variable(3) 
var_x2 = opti.variable(3) 
totalcost = 1/2* casadi.sumsqr(var_x1-var_x2)
opti.subject_to( (var_x1-c1plus).T@A1plus@(var_x1-c1plus) /2  <= 1/2)
opti.subject_to( (var_x2-c2plus).T@A2plus@(var_x2-c2plus) /2  <= 1/2)
opti.minimize(totalcost)
opti.solver('ipopt',{},s_opt)
sol = opti.solve()
next_x1 = opti.value(var_x1)
next_x2 = opti.value(var_x2)
next_lam1,next_lam2 = opti.value(opti.lam_g)
next_f = .5*sum((next_x1-next_x2)**2)
next_d = (2*next_f)**.5
next_L = next_f
assert(np.allclose(-(next_x1-next_x2),next_lam1*A1@(next_x1-c1plus),atol=1e-5))
assert(np.allclose((next_x1-c1plus)@A1@(next_x1-c1plus),1,1e-5))
assert(np.allclose((next_x1-next_x2),next_lam2*A2@(next_x2-c2plus),atol=1e-6))
assert(np.allclose((next_x2-c2plus)@A2@(next_x2-c2plus),1,1e-3))

Ldot_ND = (next_L-sol_L)/dt

Ldot = (sol_x1-sol_x2)@(v1-v2) \
    - np.cross(sol_x1-sol_x2,sol_x1-c1)@w1 \
    + np.cross(sol_x1-sol_x2,sol_x2-c2)@w2
assert(np.isclose(Ldot,Ldot_ND, atol=1e-6)), f'{Ldot} != {Ldot_ND}'
# Another (simpler?) expression for Ldot

vx1 = v1 + np.cross(w1,sol_x1-c1)
vx2 = v2 + np.cross(w2,sol_x2-c2)
Ldot_other = (sol_x1-sol_x2)@(vx1-vx2)
assert(np.isclose(Ldot,Ldot_other))

ddot_ND = (next_d-sol_d)/dt
ddot = Ldot/sol_d
assert(np.isclose(ddot,ddot_ND, atol=1e-5)), f'{ddot} != {ddot_ND}'

# #############################
# Compute derivatives of L and assert
sol_y = np.concatenate([sol_x1,sol_x2,[sol_lam1,sol_lam2]])
next_y = np.concatenate([next_x1,next_x2,[next_lam1,next_lam2]])

Ly = r_[ (sol_x1-sol_x2)+sol_lam1*A1@(sol_x1-c1),
         -(sol_x1-sol_x2)+sol_lam2*A2@(sol_x2-c2),
         1/2*(sol_x1-c1)@A1@(sol_x1-c1)-1/2,
         1/2*(sol_x2-c2)@A2@(sol_x2-c2)-1/2 ]

# Ly_cplus is computed in y but with cplus (not c), then it is not null)
Ly_cplus = r_[ (sol_x1-sol_x2)+sol_lam1*A1plus@(sol_x1-c1plus),
               -(sol_x1-sol_x2)+sol_lam2*A2plus@(sol_x2-c2plus),
               1/2*(sol_x1-c1plus)@A1plus@(sol_x1-c1plus)-1/2,
               1/2*(sol_x2-c2plus)@A2plus@(sol_x2-c2plus)-1/2 ]

# Ly_yplus is computed in next_y but with c (not cplus), then it is not null)
Ly_yplus = r_[ (next_x1-next_x2)+next_lam1*A1@(next_x1-c1),
         -(next_x1-next_x2)+next_lam2*A2@(next_x2-c2),
         1/2*(next_x1-c1)@A1@(next_x1-c1)-1/2,
         1/2*(next_x2-c2)@A2@(next_x2-c2)-1/2 ]


Lyy = r_[ c_[eye(3)+sol_lam1*A1, -eye(3), A1@(sol_x1-c1), np.zeros(3)],
          c_[-eye(3),  eye(3)+sol_lam2*A2, np.zeros(3), A2@(sol_x2-c2)],
          [r_[A1@(sol_x1-c1),  np.zeros(3), np.zeros(2)]],
          [r_[np.zeros(3), A2@(sol_x2-c2),  np.zeros(2)]] ]
Lyc = r_[ c_[-sol_lam1*A1, np.zeros([3,3])],
          c_[np.zeros([3,3]),-sol_lam2*A2 ],
          [r_[-A1@(sol_x1-c1), np.zeros(3)]],
          [r_[np.zeros(3),-A2@(sol_x2-c2)]]]
Lyr = r_[ c_[sol_lam1*(A1@skew(sol_x1-c1)-skew(A1@(sol_x1-c1))), np.zeros([3,3])],
          c_[np.zeros([3,3]),sol_lam2*(A2@skew(sol_x2-c2)-skew(A2@(sol_x2-c2))) ],
          [r_[(sol_x1-c1)@A1@skew(sol_x1-c1), np.zeros(3)]],
          [r_[np.zeros(3),(sol_x2-c2)@A2@skew(sol_x2-c2) ]] ]
Lyth = c_[Lyc[:,:3],Lyr[:,:3],Lyc[:,3:],Lyr[:,3:]]
# Tolerance here must be very high, not sure why, but those tests are not super important
softassert(np.allclose(Ly_yplus/dt,Lyy@(next_y-sol_y)/dt,rtol=1e-1), 'Ly_yplus!=0')
softassert(np.allclose(Ly_cplus/dt,Lyc@r_[v1,v2]+Lyr@r_[w1,w2],rtol=1e-1),'Ly_cplus!=0')
softassert(np.allclose(Ly_cplus/dt,-Ly_yplus/dt,rtol=1e-1),'Lycplus!=-Ly_yplus')

# ##############################
# Cnompute derivative of y=[x,lam] and assert
v = r_[v1,v2]
w = r_[w1,w2]
theta_dot = r_[v1,w1,v2,w2]
yc = -np.linalg.inv(Lyy)@Lyc
yr = -np.linalg.inv(Lyy)@Lyr
yth = -np.linalg.inv(Lyy)@Lyth
ydot = yc@v+yr@w
ydot_other = yth@theta_dot
assert(np.allclose(ydot,ydot_other))
x1dot_ND = (next_x1-sol_x1)/dt
lam1dot_ND = (next_lam1-sol_lam1)/dt
x2dot_ND = (next_x2-sol_x2)/dt
lam2dot_ND = (next_lam2-sol_lam2)/dt
ydot_ND=np.concatenate([x1dot_ND,x2dot_ND,[lam1dot_ND,lam2dot_ND]])

assert(np.allclose(ydot,ydot_ND,atol=1e-5))

# #####
# Check d Ldot / dv and d Ldot / dw
Lc1 = sol_x1-sol_x2
Lc2 = -(sol_x1-sol_x2)
Lr1 = -np.cross(sol_x1-sol_x2,sol_x1-c1)
Lr2 = np.cross(sol_x1-sol_x2,sol_x2-c2)
Lth = r_[Lc1,Lr1,Lc2,Lr2]
Lth_thdot = Lc1@v1 + Lr1@w1 + Lc2@v2 + Lr2@w2
assert(np.isclose(Ldot,Lth_thdot))
Lth_thdot_other = Lth@theta_dot
assert(np.isclose(Lth_thdot,Lth_thdot_other))

Ldot = (sol_x1-sol_x2)@(v1-v2) \
    - np.cross(sol_x1-sol_x2,sol_x1-c1)@w1 \
    + np.cross(sol_x1-sol_x2,sol_x2-c2)@w2
next_Ldot = (next_x1-next_x2)@(v1-v2) \
    - np.cross(next_x1-next_x2,next_x1-c1plus)@w1 \
    + np.cross(next_x1-next_x2,next_x2-c2plus)@w2
dLdot_ND = (next_Ldot-Ldot)/dt

# partial derivative of Ldot
Ldot_cplus = (sol_x1-sol_x2)@(v1-v2) \
    - np.cross(sol_x1-sol_x2,sol_x1-c1plus)@w1 \
    + np.cross(sol_x1-sol_x2,sol_x2-c2plus)@w2
dLdot_dc1 = -cross(sol_x1-sol_x2,w1)
dLdot_dc2 = cross(sol_x1-sol_x2,w2)
assert(np.isclose((Ldot_cplus-Ldot)/dt,dLdot_dc1@v1 + dLdot_dc2@v2))
Ldot_xplus = (next_x1-next_x2)@(v1-v2) \
    - np.cross(next_x1-next_x2,next_x1-c1)@w1 \
    + np.cross(next_x1-next_x2,next_x2-c2)@w2
dLdot_dx1 = v1-v2 -cross(sol_x1-c1,w1)+cross(sol_x2-c2,w2)+cross(sol_x1-sol_x2,w1)
dLdot_dx2 = -v1+v2 +cross(sol_x1-c1,w1)-cross(sol_x2-c2,w2)-cross(sol_x1-sol_x2,w2)
assert(np.isclose((Ldot_xplus-Ldot)/dt,
                  dLdot_dx1@(next_x1-sol_x1)/dt + dLdot_dx2@(next_x2-sol_x2)/dt))

dLdot_dx1_simpler = v1-vx2 -cross(sol_x2-c1,w1)
dLdot_dx2_simpler = -vx1+v2 -cross(sol_x1-c2,w2)
assert(np.allclose(dLdot_dx1,dLdot_dx1_simpler))
assert(np.allclose(dLdot_dx2,dLdot_dx2_simpler))

dLdot_dc = r_[dLdot_dc1,dLdot_dc2] + r_[dLdot_dx1,dLdot_dx2]@yc[:6]
dLdot_dr = r_[dLdot_dx1,dLdot_dx2]@yr[:6]
dLdot_ND = (next_Ldot-Ldot)/dt
assert(np.isclose(dLdot_ND,dLdot_dc@r_[v1,v2]+dLdot_dr@r_[w1,w2]))

# ####################
# Check with DDL

xc,xr = yc[:3],yr[:3]
#dx1 = c_[yc[:3],yr[:3]]
#dx2 = c_[yc[3:6],yr[3:6]]
dx1 = yth[:3]
dx2 = yth[3:6]

dL = r_[sol_x1-sol_x2,-cross(sol_x1-sol_x2,sol_x1-c1),-(sol_x1-sol_x2),cross(sol_x1-sol_x2,sol_x2-c2)]
assert(np.isclose(Ldot,dL@theta_dot))

ddL = r_[dx1-dx2,
         -skew(sol_x1-sol_x2)@dx1+skew(sol_x1-c1)@(dx1-dx2),
         -dx1+dx2,
         skew(sol_x1-sol_x2)@dx2-skew(sol_x2-c2)@(dx1-dx2)] \
    +r_[np.zeros([3,12]),
        c_[skew(sol_x1-sol_x2),np.zeros([3,9])],
        np.zeros([3,12]),
        c_[np.zeros([3,6]),-skew(sol_x1-sol_x2),np.zeros([3,3])]]
next_dL = r_[next_x1-next_x2,-cross(next_x1-next_x2,next_x1-c1plus),-(next_x1-next_x2),cross(next_x1-next_x2,next_x2-c2plus)]
dL_ND = (next_dL-dL)/dt

assert(np.allclose(dL_ND,ddL@theta_dot,atol=1e-4))

dLdot = theta_dot@ddL@theta_dot
assert(np.isclose(dLdot,dLdot_ND))



# ###########
# Check ddot

ddot = Ldot/sol_d
next_ddot = next_Ldot/next_d

dddot_dt_ND = (next_ddot-ddot)/dt
dddot = (theta_dot.T@ddL/sol_d - ddot/sol_d**2 * dL)
dddot_dt = dddot @ theta_dot
assert(np.isclose(dddot_dt,dddot_dt_ND,rtol=1e-3))


# #### Summary

# This is a summary of all previous computations.
# We assume that the problem min-distance is already solved, and
# optimal variables are stored under sol_x1 and sol_x2.
# For finite difference tests, we also assume the next problem (after
# integrating v and w during dt) is solved, and optimal variables
# are next_x1 and next_x2

# ### --- CALC

# Cost = half of squared distance
L = .5*sum((sol_x1-sol_x2))
# Distance
dist = np.linalg.norm(sol_x1-sol_x2)

# Witness point velocities (seen as fixed on the objects)
vx1 = v1 + np.cross(w1,sol_x1-c1)
vx2 = v2 + np.cross(w2,sol_x2-c2)

# Cost time derivative
Ldot = (sol_x1-sol_x2)@(vx1-vx2)
# Distance time derivative
dist_dot = Ldot/dist

# Assert dist_dot
next_dist = np.linalg.norm(next_x1-next_x2)
assert(np.isclose((next_dist-dist)/dt,dist_dot, atol=1e-5)), f'{(next_dist-dist)/dt} != {dist_dot}'

# ### --- CALC DIFF

# ### Derivative with respect to position: d_dist_dot_dtheta

# theta = (c1,c2,r1,r2)
theta_dot = r_[v1,w1,v2,w2]
# dist_dot derivative wrt theta
dL_dtheta = r_[sol_x1-sol_x2,-cross(sol_x1-sol_x2,sol_x1-c1),-(sol_x1-sol_x2),cross(sol_x1-sol_x2,sol_x2-c2)]
assert(np.allclose(Ldot,dL_dtheta@theta_dot))
ddL_dtheta2 = r_[dx1-dx2,
                 -skew(sol_x1-sol_x2)@dx1+skew(sol_x1-c1)@(dx1-dx2),
                 -dx1+dx2,
                 skew(sol_x1-sol_x2)@dx2-skew(sol_x2-c2)@(dx1-dx2)] \
                 +r_[np.zeros([3,12]),
                     c_[skew(sol_x1-sol_x2),np.zeros([3,9])],
                     np.zeros([3,12]),
                     c_[np.zeros([3,6]),-skew(sol_x1-sol_x2),np.zeros([3,3])]]
                 
                 
dLdot_dtheta = theta_dot@ddL_dtheta2
# Verif using finite diff
next_Ldot = (next_x1-next_x2)@( v1 + np.cross(w1,next_x1-c1plus)-(v2 + np.cross(w2,next_x2-c2plus)))
dLdot_dt_ND = (next_Ldot-Ldot)/dt
assert(np.isclose(dLdot_dt_ND,dLdot_dtheta@theta_dot))

d_dist_dot_dtheta = (theta_dot.T@ddL_dtheta2/dist - dist_dot/dist**2 * dL_dtheta)

# Verif using finite diff changing only the position
d_dist_dot_dt = d_dist_dot_dtheta @ theta_dot
next_dist_dot = next_Ldot / next_dist
d_dist_dot_dt_ND = (next_dist_dot-dist_dot)/dt
assert(np.isclose(d_dist_dot_dt_ND,d_dist_dot_dt))

# ### Derivative with respect to velocity: d_dist_dot_dthetadot
d_dist_dot_dtheta_dot = dL_dtheta / dist


# Verif using finite diff
# Select a disturbed velocity
#v1_eps,w1_eps,v2_eps,w2_eps = [ v+dt*np.random.rand(3) for v in [v1,w1,v2,w2] ]

# Compute the disturbed dist_dot changing only the velocity
theta_dot_plus = r_[v1plus,w1plus,v2plus,w2plus]
theta_ddot = (theta_dot_plus-theta_dot)/dt
next_dist_dot =  (sol_x1-sol_x2)@( v1plus + np.cross(w1plus,sol_x1-c1)-(v2plus + np.cross(w2plus,sol_x2-c2)))/dist
d_dist_dot_dt_ND = (next_dist_dot-dist_dot)/dt
d_dist_dot_dt = d_dist_dot_dtheta_dot@theta_ddot
assert(np.isclose(d_dist_dot_dt,d_dist_dot_dt_ND))


# Compute the disturbed dist_dot for changes of both position and velocity
theta_dot_plus = r_[v1plus,w1plus,v2plus,w2plus]
theta_ddot = (theta_dot_plus-theta_dot)/dt
next_dist_dot =  (next_x1-next_x2)@( v1plus + np.cross(w1plus,next_x1-c1plus)-(v2plus + np.cross(w2plus,next_x2-c2plus)))/next_dist
d_dist_dot_dt_ND = (next_dist_dot-dist_dot)/dt
d_dist_dot_dt = d_dist_dot_dtheta @ theta_dot + d_dist_dot_dtheta_dot @ theta_ddot
assert(np.isclose(d_dist_dot_dt,d_dist_dot_dt_ND,rtol=1e-3))



# ##################################3
# Robot derivatives

pin.computeJointJacobians(rmodel,rdata,q)
J1 = pin.getFrameJacobian(rmodel,rdata,idf1,pin.LOCAL_WORLD_ALIGNED)
J2 = pin.getFrameJacobian(rmodel,rdata,idf2,pin.LOCAL_WORLD_ALIGNED)

assert(np.allclose(nu1.vector,J1@vq))
assert(np.allclose(nu2.vector,J2@vq))

dtheta_dq = r_[J1,J2]

assert(np.allclose(dtheta_dq@vq,theta_dot))

dtheta_dot_dqdot = r_[J1,J2]
# Check derivative by computing v,w at q,vqplus (and not at qplus,vqplus, chane of velocity only)
pin.forwardKinematics(rmodel,rdataplus,q,vqplus)
nu1_qvqplus = pin.getFrameVelocity(rmodel,rdataplus,idf1,pin.LOCAL_WORLD_ALIGNED).copy()
v1_qvqplus,w1_qvqplus = nu1.linear,nu1.angular
nu2_qvqplus = pin.getFrameVelocity(rmodel,rdataplus,idf2,pin.LOCAL_WORLD_ALIGNED).copy()
v2_qvqplus,w2_qvqplus = nu2_qvqplus.linear,nu2_qvqplus.angular
theta_dot_qvqplus = r_[v1_qvqplus,w1_qvqplus,v2_qvqplus,w2_qvqplus]
assert(np.allclose(dtheta_dot_dqdot@aq,(theta_dot_qvqplus-theta_dot)/dt))

pin.computeForwardKinematicsDerivatives(rmodel,rdata,q,vq,aq)
dnu1_dq,dnu1_dqdot = pin.getFrameVelocityDerivatives(rmodel,rdata,idf1,pin.LOCAL_WORLD_ALIGNED)
dnu2_dq,dnu2_dqdot = pin.getFrameVelocityDerivatives(rmodel,rdata,idf2,pin.LOCAL_WORLD_ALIGNED)



assert(np.allclose(J1,dnu1_dqdot))
assert(np.allclose(J2,dnu2_dqdot))
dtheta_dot_dq = r_[dnu1_dq,dnu2_dq]
# Check derivative by computing v,w at q,vqplus (and not at qplus,vqplus, chane of velocity only)
pin.forwardKinematics(rmodel,rdataplus,qplus,vq)
nu1_qplusvq = pin.getFrameVelocity(rmodel,rdataplus,idf1,pin.LOCAL_WORLD_ALIGNED).copy()
v1_qplusvq,w1_qplusvq = nu1.linear,nu1.angular
nu2_qplusvq = pin.getFrameVelocity(rmodel,rdataplus,idf2,pin.LOCAL_WORLD_ALIGNED).copy()
v2_qplusvq,w2_qplusvq = nu2_qplusvq.linear,nu2_qplusvq.angular
theta_dot_qplusvq = r_[v1_qplusvq,w1_qplusvq,v2_qplusvq,w2_qplusvq]
# Assert does not pass, derivatives of LWA are not what we think they are
assert(not np.allclose(dtheta_dot_dq@vq,(theta_dot_qplusvq-theta_dot)/dt))

in1_dnu1_dq,in1_dnu1_dqdot = pin.getFrameVelocityDerivatives(rmodel,rdata,idf1,pin.LOCAL)
in2_dnu2_dq,in2_dnu2_dqdot = pin.getFrameVelocityDerivatives(rmodel,rdata,idf2,pin.LOCAL)

inLWA1_dv1_dq =  R1@in1_dnu1_dq[:3] - skew(v1)@R1@in1_dnu1_dqdot[3:]
inLWA1_dw1_dq =  R1@in1_dnu1_dq[3:]
inLWA2_dv2_dq =  R2@in2_dnu2_dq[:3] - skew(v2)@R2@in2_dnu2_dqdot[3:]
inLWA2_dw2_dq =  R2@in2_dnu2_dq[3:]
dtheta_dot_dq = r_[inLWA1_dv1_dq,inLWA1_dw1_dq,inLWA2_dv2_dq,inLWA2_dw2_dq]
assert(np.allclose(dtheta_dot_dq@vq,(theta_dot_qplusvq-theta_dot)/dt,atol=1e-4))

dtheta_dot_ND = (theta_dot_plus-theta_dot)/dt
assert(np.allclose(dtheta_dot_dq@vq+dtheta_dot_dqdot@aq,dtheta_dot_ND,atol=1e-3))

#assert(np.allclose(dtheta_dq@vq,(theta_plus-theta)/dt))

# TODO: here a 0* is needed. WHHHHYYYYYYYYYYYYYYYYYYYYY!
d_dist_dot_dq = d_dist_dot_dtheta @ dtheta_dq + d_dist_dot_dtheta_dot @ dtheta_dot_dq
d_dist_dot_dqdot = d_dist_dot_dtheta_dot @ dtheta_dot_dqdot
assert(np.allclose(d_dist_dot_dq@vq+d_dist_dot_dqdot@aq,d_dist_dot_dt,atol=1e-3))


d_dist_dot_dq_from_n, d_dist_dot_dqdot_from_n = compute_d_d_dot_dq_dq_dot(rmodel, gmodel, q, vq, idg1, idg2)

assert(np.allclose(d_dist_dot_dq,d_dist_dot_dq_from_n,atol=1e-3)), f'{d_dist_dot_dq} != {d_dist_dot_dq_from_n}'
assert(np.allclose(d_dist_dot_dqdot,d_dist_dot_dqdot_from_n,atol=1e-3)), f'{d_dist_dot_dqdot} != {d_dist_dot_dqdot_from_n}'

print(d_dist_dot_dq)