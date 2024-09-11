import pinocchio as pin
import numpy as np
import hppfcl
from wrapper_panda import PandaWrapper

def add_ellipsoid(
    cmodel: pin.GeometryModel,
    name: str,
    parentJoint=0,
    parentFrame=0,
    placement=pin.SE3.Random(),
    dim=[0.2, 0.5, 0.1],
) -> pin.GeometryModel:
    """
    Add an ellipsoid geometry object to the given `cmodel`.

    Args:
        cmodel (pin.GeometryModel): The geometry model to add the ellipsoid to.
        name (str): The name of the ellipsoid.
        parentJoint (int, optional): The index of the parent joint. Defaults to 0.
        parentFrame (int, optional): The index of the parent frame. Defaults to 0.
        placement (pin.SE3, optional): The placement of the ellipsoid. Defaults to pin.SE3.Random().
        dim (List[float], optional): The dimensions of the ellipsoid [x, y, z]. Defaults to [0.2, 0.5, 0.1].

    Returns:
        pin.GeometryModel: The updated geometry model with the added ellipsoid.
    """

    elips = hppfcl.Ellipsoid(dim[0], dim[1], dim[2])
    elips_geom = pin.GeometryObject(
        name,
        parent_joint=parentJoint,
        parent_frame=parentFrame,
        collision_geometry=elips,
        placement=placement,
    )
    elips_geom.meshColor = np.concatenate(
        (np.random.uniform(0, 1, 3), np.ones(1) / 0.8)
    )

    cmodel.addGeometryObject(elips_geom)
    return cmodel


def c1(q):
    pin.forwardKinematics(rmodel, rdata, q)
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    c = cdata.oMg[id_rob].translation
    return c

def numdiff_matrix(f, inX, h=1e-6):
    f0 = np.array(f(inX)).copy()
    x = inX.copy()
    df_dx = np.zeros((f0.size, len(x)))
    for ix in range(len(x)):
        x[ix] += h
        df_dx[:, ix] = (f(x) - f0) / h
        x[ix] = inX[ix]
    return df_dx


wrapper_panda = PandaWrapper()
rmodel, cmodel, vmodel = wrapper_panda()

dim = [0.2, 0.5, 0.1]

elips = hppfcl.Ellipsoid(dim[0], dim[1], dim[2])
elips_geom = pin.GeometryObject(
    "obstacle",
    parent_joint=0,
    parent_frame=0,
    collision_geometry=elips,
    placement=pin.SE3.Random(),
)
id_obs = cmodel.addGeometryObject(elips_geom)

elips = hppfcl.Ellipsoid(dim[0], dim[1], dim[2])
elips_geom = pin.GeometryObject(
    "robot",
    parent_joint=cmodel.geometryObjects[
    cmodel.getGeometryId("panda2_link7_sc_5")
].parentJoint,
    parent_frame=cmodel.geometryObjects[
    cmodel.getGeometryId("panda2_link7_sc_5")
].parentFrame,
    collision_geometry=elips,
    placement=pin.SE3.Random(),
)

id_rob = cmodel.addGeometryObject(elips_geom)

rdata = rmodel.createData()
cdata = cmodel.createData()

q = pin.randomConfiguration(rmodel)

dtheta1_dq = pin.computeFrameJacobian(rmodel, rdata, q, cmodel.geometryObjects[
    cmodel.getGeometryId("panda2_link7_sc_5")
].parentFrame, pin.LOCAL_WORLD_ALIGNED)

dc1_dq = dtheta1_dq[:3][:]
dc1_dq_nd = numdiff_matrix(c1, q)

print("diff",dc1_dq - dc1_dq_nd)

print("ana",dc1_dq)
print("numdiff",dc1_dq_nd)

