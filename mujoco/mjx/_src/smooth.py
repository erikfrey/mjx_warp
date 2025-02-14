import warp as wp

from . import math
from . import types


def kinematics(m: types.Model, d: types.Data):
  """Forward kinematics."""

  @wp.kernel
  def _root(m: types.Model, d: types.Data):
    worldid = wp.tid()
    d.xpos[worldid, 0] = wp.vec3(0.0)
    d.xquat[worldid, 0] = wp.quat(1.0, 0.0, 0.0, 0.0)
    d.xipos[worldid, 0] = wp.vec3(0.0)
    d.xmat[worldid, 0] = wp.identity(n=3, dtype=wp.float32)
    d.ximat[worldid, 0] = wp.identity(n=3, dtype=wp.float32)

  @wp.kernel
  def _level(m: types.Model, d: types.Data, leveladr: int):
    worldid, nodeid = wp.tid()
    bodyid = m.body_tree[leveladr + nodeid]
    jntadr = m.body_jntadr[bodyid]
    jntnum = m.body_jntnum[bodyid]
    qpos = d.qpos[worldid]

    if jntnum == 1 and m.jnt_type[jntadr] == 0:
      # free joint
      qadr = m.jnt_qposadr[jntadr]
      # TODO(erikfrey): would it be better to use some kind of wp.copy here?
      xpos = wp.vec3(qpos[qadr], qpos[qadr + 1], qpos[qadr + 2])
      xquat = wp.quat(
          qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6]
      )
      d.xanchor[worldid, jntadr] = xpos
      d.xaxis[worldid, jntadr] = m.jnt_axis[jntadr]
    else:
      # regular or no joints
      # apply fixed translation and rotation relative to parent
      pid = m.body_parentid[bodyid]
      xpos = (d.xmat[worldid, pid] * m.body_pos[bodyid]) + d.xpos[worldid, pid]
      xquat = math.mul_quat(d.xquat[worldid, pid], m.body_quat[bodyid])

      for _ in range(jntnum):
        qadr = m.jnt_qposadr[jntadr]
        jnt_type = m.jnt_type[jntadr]
        xanchor = math.rot_vec_quat(m.jnt_pos[jntadr], xquat) + xpos
        xaxis = math.rot_vec_quat(m.jnt_axis[jntadr], xquat)

        if jnt_type == 3:  # hinge
          qloc = math.axis_angle_to_quat(
              m.jnt_axis[jntadr], d.qpos[worldid, qadr] - m.qpos0[qadr]
          )
          xquat = math.mul_quat(xquat, qloc)
          # correct for off-center rotation
          xpos = xanchor - math.rot_vec_quat(m.jnt_pos[jntadr], xquat)

        d.xanchor[worldid, jntadr] = xanchor
        d.xaxis[worldid, jntadr] = xaxis
        jntadr += 1

    d.xpos[worldid, bodyid] = xpos
    d.xquat[worldid, bodyid] = wp.normalize(xquat)
    d.xmat[worldid, bodyid] = math.quat_to_mat(xquat)

  wp.launch(_root, dim=(d.nworld), inputs=[m, d])
  for adr, size in zip(m.body_leveladr.numpy(), m.body_levelsize.numpy()):
    wp.launch(_level, dim=(d.nworld, size), inputs=[m, d, adr])


def com_pos(m: types.Model, d: types.Data):
  """Map inertias and motion dofs to global frame centered at subtree-CoM."""

  @wp.kernel
  def mass_subtree_acc(
      m: types.Model,
      mass_subtree: wp.array(dtype=wp.float32, ndim=1),
      leveladr: int,
  ):
    nodeid = wp.tid()
    bodyid = m.body_tree[leveladr + nodeid]
    pid = m.body_parentid[bodyid]
    wp.atomic_add(mass_subtree, pid, mass_subtree[bodyid])

  @wp.kernel
  def subtree_com_init(m: types.Model, d: types.Data):
    worldid, bodyid = wp.tid()
    d.subtree_com[worldid, bodyid] = (
        d.xipos[worldid, bodyid] * m.body_mass[bodyid]
    )

  @wp.kernel
  def subtree_com_acc(m: types.Model, d: types.Data, leveladr: int):
    worldid, nodeid = wp.tid()
    bodyid = m.body_tree[leveladr + nodeid]
    pid = m.body_parentid[bodyid]
    wp.atomic_add(d.subtree_com, worldid, pid, d.subtree_com[worldid, bodyid])

  @wp.kernel
  def subtree_div(
      mass_subtree: wp.array(dtype=wp.float32, ndim=1), d: types.Data
  ):
    worldid, bodyid = wp.tid()
    d.subtree_com[worldid, bodyid] /= mass_subtree[bodyid]

  @wp.kernel
  def cinert(m: types.Model, d: types.Data):
    worldid, bodyid = wp.tid()
    mat = d.ximat[worldid, bodyid]
    inert = m.body_inertia[bodyid]
    mass = m.body_mass[bodyid]
    dif = (
        d.xipos[worldid, bodyid] - d.subtree_com[worldid, m.body_rootid[bodyid]]
    )
    # express inertia in com-based frame (mju_inertCom)

    res = types.vec10()
    # res_rot = mat * diag(inert) * mat'
    tmp = mat @ wp.diag(inert) @ wp.transpose(mat)
    res[0] = tmp[0, 0]
    res[1] = tmp[1, 1]
    res[2] = tmp[2, 2]
    res[3] = tmp[0, 1]
    res[4] = tmp[0, 2]
    res[5] = tmp[1, 2]
    # res_rot -= mass * dif_cross * dif_cross
    res[0] += mass * (dif[1] * dif[1] + dif[2] * dif[2])
    res[1] += mass * (dif[0] * dif[0] + dif[2] * dif[2])
    res[2] += mass * (dif[0] * dif[0] + dif[1] * dif[1])
    res[3] -= mass * dif[0] * dif[1]
    res[4] -= mass * dif[0] * dif[2]
    res[5] -= mass * dif[1] * dif[2]
    # res_tran = mass * dif
    res[6] = mass * dif[0]
    res[7] = mass * dif[1]
    res[8] = mass * dif[2]
    # res_mass = mass
    res[9] = mass

    d.cinert[worldid, bodyid] = res

  @wp.kernel
  def cdof(m: types.Model, d: types.Data):
    worldid, jntid = wp.tid()
    bodyid = m.jnt_bodyid[jntid]
    dofid = m.jnt_dofadr[jntid]
    jnt_type = m.jnt_type[jntid]
    xaxis = d.xaxis[worldid, jntid]
    xmat = wp.transpose(d.xmat[worldid, bodyid])

    # compute com-anchor vector
    offset = (
        d.subtree_com[worldid, m.body_rootid[bodyid]]
        - d.xanchor[worldid, jntid]
    )

    res = d.cdof[worldid]
    if jnt_type == 0:  # free
      res[dofid + 0] = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
      res[dofid + 1] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
      res[dofid + 2] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
      # I_3 rotation in child frame (assume no subsequent rotations)
      res[dofid + 3] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))
      res[dofid + 4] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))
      res[dofid + 5] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))
    elif jnt_type == 1:  # ball
      # I_3 rotation in child frame (assume no subsequent rotations)
      res[dofid + 0] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))
      res[dofid + 1] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))
      res[dofid + 2] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))
    elif jnt_type == 2:  # slide
      res[dofid] = wp.spatial_vector(wp.vec3(0.0), xaxis)
    elif jnt_type == 3:  # hinge
      res[dofid] = wp.spatial_vector(xaxis, wp.cross(xaxis, offset))

  leveladr, levelsize = m.body_leveladr.numpy(), m.body_levelsize.numpy()

  mass_subtree = wp.clone(m.body_mass)
  for i in range(len(leveladr) - 1, -1, -1):
    adr, size = leveladr[i], levelsize[i]
    wp.launch(mass_subtree_acc, dim=(size,), inputs=[m, mass_subtree, adr])

  wp.launch(subtree_com_init, dim=(d.nworld, m.nbody), inputs=[m, d])

  for i in range(len(leveladr) - 1, -1, -1):
    adr, size = leveladr[i], levelsize[i]
    wp.launch(subtree_com_acc, dim=(d.nworld, size), inputs=[m, d, adr])

  wp.launch(subtree_div, dim=(d.nworld, m.nbody), inputs=[mass_subtree, d])
  wp.launch(cinert, dim=(d.nworld, m.nbody), inputs=[m, d])
  wp.launch(cdof, dim=(d.nworld, m.njnt), inputs=[m, d])


def crb(m: types.Model, d: types.Data):
  """Composite rigid body inertia algorithm."""

  wp.copy(d.crb, d.cinert)

  @wp.kernel
  def crb_accumulate(m: types.Model, d: types.Data, leveladr: int):
    worldid, nodeid = wp.tid()
    bodyid = m.body_tree[leveladr + nodeid]
    pid = m.body_parentid[bodyid]
    if pid == 0:
      return
    wp.atomic_add(d.crb, worldid, pid, d.crb[worldid, bodyid])

  @wp.kernel
  def qM_sparse(m: types.Model, d: types.Data):
    worldid, dofid = wp.tid()
    madr_ij = m.dof_Madr[dofid]
    bodyid = m.dof_bodyid[dofid]

    # init M(i,i) with armature inertia
    d.qM[worldid, 0, madr_ij] = m.dof_armature[dofid]

    # precompute buf = crb_body_i * cdof_i
    i = d.crb[worldid, bodyid]
    v = d.cdof[worldid, dofid]
    # multiply 6D vector (rotation, translation) by 6D inertia matrix (mju_mulInertVec)
    buf = wp.spatial_vector()
    buf[0] = i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5]
    buf[1] = i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5]
    buf[2] = i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4]
    buf[3] = i[8] * v[1] - i[7] * v[2] + i[9] * v[3]
    buf[4] = i[6] * v[2] - i[8] * v[0] + i[9] * v[4]
    buf[5] = i[7] * v[0] - i[6] * v[1] + i[9] * v[5]
    # sparse backward pass over ancestors
    while dofid >= 0:
      d.qM[worldid, 0, madr_ij] += wp.dot(d.cdof[worldid, dofid], buf)
      madr_ij += 1
      dofid = m.dof_parentid[dofid]

  leveladr, levelsize = m.body_leveladr.numpy(), m.body_levelsize.numpy()

  for i in range(len(leveladr) - 1, -1, -1):
    adr, size = leveladr[i], levelsize[i]
    wp.launch(crb_accumulate, dim=(d.nworld, size), inputs=[m, d, adr])

  d.qM.zero_()
  wp.launch(qM_sparse, dim=(d.nworld, m.nv), inputs=[m, d])


def _factor_m_sparse(m: types.Model, d: types.Data, qM: wp.array, qLD: wp.array, qLDiagInv: wp.array):
  """Sparse L'*D*L factorizaton of inertia-like matrix M, assumed spd."""

  @wp.kernel
  def qLD_acc(m: types.Model, qLD: wp.array3d(dtype=wp.float32), leveladr: int):
    worldid, nodeid = wp.tid()
    update = m.qLD_updates[leveladr + nodeid]
    i, k, Madr_ki = update[0], update[1], update[2]
    Madr_i = m.dof_Madr[i]
    # tmp = M(k,i) / M(k,k)
    tmp = qLD[worldid, 0, Madr_ki] / qLD[worldid, 0, m.dof_Madr[k]]
    for j in range(m.dof_Madr[i + 1] - Madr_i):
      # M(i,j) -= M(k,j) * tmp
      wp.atomic_sub(qLD[worldid, 0], Madr_i + j, qLD[worldid, 0, Madr_ki + j] * tmp)
    # M(k,i) = tmp
    qLD[worldid, 0, Madr_ki] = tmp

  @wp.kernel
  def qLDiag_div(m: types.Model, qLD: wp.array3d(dtype=wp.float32), qLDiagInv: wp.array2d(dtype=wp.float32)):
    worldid, dofid = wp.tid()
    qLDiagInv[worldid, dofid] = 1.0 / qLD[worldid, 0, m.dof_Madr[dofid]]

  wp.copy(qLD, qM)

  leveladr, levelsize = m.qLD_leveladr.numpy(), m.qLD_levelsize.numpy()

  for i in range(len(leveladr) - 1, -1, -1):
    adr, size = leveladr[i], levelsize[i]
    wp.launch(qLD_acc, dim=(d.nworld, size), inputs=[m, qLD, adr])

  wp.launch(qLDiag_div, dim=(d.nworld, m.nv), inputs=[m, qLD, qLDiagInv])

def _factor_m_dense(m: types.Model, d: types.Data, qM: wp.array, qLD: wp.array, block_dim: int = 32):
  """Dense Cholesky factorizaton of inertia-like matrix M, assumed spd."""

  TILE = m.nv
  BLOCK_DIM = block_dim

  @wp.kernel
  def cholesky(qM: wp.array3d(dtype=wp.float32), qLD: wp.array3d(dtype=wp.float32)):
    worldid = wp.tid()
    qM_tile = wp.tile_load(qM[worldid], shape=(TILE, TILE))
    qLD_tile = wp.tile_cholesky(qM_tile)
    wp.tile_store(qLD[worldid], qLD_tile)

  wp.launch_tiled(cholesky, dim=(d.nworld), inputs=[qM, qLD], block_dim=BLOCK_DIM)


def factor_m(m: types.Model, d: types.Data, qM: wp.array3d, qLD: wp.array3d, qLDiagInv: wp.array = None, block_dim: int = 32):
  """Factorizaton of inertia-like matrix M, assumed spd."""

  if wp.static(m.opt.is_sparse):
    assert qLDiagInv is not None
    _factor_m_sparse(m, d, qM, qLD, qLDiagInv)
  else:
    _factor_m_dense(m, d, qM, qLD, block_dim=block_dim)

def solve_m(
    m: types.Model, d: types.Data, qLD: wp.array, qLDiagInv: wp.array, x: wp.array2d(dtype=wp.float32), y: wp.array2d(dtype=wp.float32), block_dim: int = 32
):
  """Computes sparse backsubstitution:  x = inv(L'*D*L)*y ."""

  TILE = m.nv
  BLOCK_DIM = block_dim

  @wp.kernel
  def solve_m_dense(
      qLD: wp.array3d(dtype=wp.float32), x: wp.array2d(dtype=wp.float32), y: wp.array2d(dtype=wp.float32)
  ):
    worldid = wp.tid()
    qLD_tile = wp.tile_load(qLD[worldid], shape=(TILE, TILE))
    x_tile = wp.tile_load(x[worldid], shape=(TILE))
    y_tile = wp.tile_cholesky_solve(qLD_tile, x_tile)
    wp.tile_store(y[worldid], y_tile)

  @wp.kernel
  def solve_m_sparse(
      m: types.Model, qLD: wp.array3d(dtype=wp.float32), qLDiagInv: wp.array2d(dtype=wp.float32), y: wp.array2d(dtype=wp.float32)
  ):
    worldid = wp.tid()

    # x <- inv(L') * x;
    for i in range(m.nv-1, -1, -1):
      madr_ij = m.dof_Madr[i] + 1
      j = m.dof_parentid[i]

      while j >= 0:
        y[worldid, j] = y[worldid, j] - qLD[worldid, 0, madr_ij] * y[worldid, i]
        madr_ij += 1
        j = m.dof_parentid[j]

    # x <- inv(D) * x
    for i in range(m.nv):
      y[worldid, i] = y[worldid, i] * qLDiagInv[worldid, i]

    # x <- inv(L) * x; 
    for i in range(m.nv):
      madr_ij = m.dof_Madr[i] + 1
      j = m.dof_parentid[i]

      while j >= 0:
        y[worldid, i] = y[worldid, i] - qLD[worldid, 0, madr_ij] * y[worldid, j]
        madr_ij += 1
        j = m.dof_parentid[j]

  if (m.opt.is_sparse):
    wp.copy(y, x)
    wp.launch(solve_m_sparse, dim=(d.nworld), inputs=[m, qLD, qLDiagInv, y])
  else:
    wp.launch_tiled(solve_m_dense, dim=(d.nworld), inputs=[qLD, x, y], block_dim=BLOCK_DIM)

def rne(m: types.Model, d: types.Data):
  """Computes inverse dynamics using Newton-Euler algorithm."""

  cacc = wp.zeros(shape=(d.nworld, m.nbody), dtype=wp.spatial_vector)
  cfrc = wp.zeros(shape=(d.nworld, m.nbody), dtype=wp.spatial_vector)

  @wp.kernel
  def cacc_gravity(m: types.Model, cacc: wp.array(dtype=wp.spatial_vector, ndim=2)):
    worldid = wp.tid()
    cacc[worldid, 0] = wp.spatial_vector(wp.vec3(0.0), -m.opt.gravity)

  @wp.kernel
  def cacc_level(m: types.Model, d: types.Data, cacc: wp.array(dtype=wp.spatial_vector, ndim=2), leveladr: int):
    worldid, nodeid = wp.tid()
    bodyid = m.body_tree[leveladr + nodeid]
    dofnum = m.body_dofnum[bodyid]
    pid = m.body_parentid[bodyid]
    dofadr = m.body_dofadr[bodyid]
    local_cacc = cacc[worldid, pid]
    for i in range(dofnum):
      local_cacc += d.cdof_dot[worldid, dofadr + i] * d.qvel[worldid, dofadr + i]
    cacc[worldid, bodyid] = local_cacc

  @wp.kernel
  def frc(d: types.Data, cfrc: wp.array(dtype=wp.spatial_vector, ndim=2), cacc: wp.array(dtype=wp.spatial_vector, ndim=2)):
    worldid, bodyid = wp.tid()
    tmp0 = math.inert_vec(d.cinert[worldid, bodyid], cacc[worldid, bodyid])
    tmp1 = math.inert_vec(d.cinert[worldid, bodyid], d.cvel[worldid, bodyid])
    tmp2 = math.motion_cross_force(d.cvel[worldid, bodyid], tmp1)
    cfrc[worldid, bodyid] += tmp0 + tmp2

  @wp.kernel
  def cfrc_fn(m: types.Model, cfrc: wp.array(dtype=wp.spatial_vector, ndim=2), leveladr: int):
    worldid, nodeid = wp.tid()
    bodyid = m.body_tree[leveladr + nodeid]
    pid = m.body_parentid[bodyid]
    wp.atomic_add(cfrc[worldid], pid, cfrc[worldid, bodyid])

  @wp.kernel
  def qfrc_bias(m: types.Model, d: types.Data, cfrc: wp.array(dtype=wp.spatial_vector, ndim=2)):
    worldid, dofid = wp.tid()
    bodyid = m.dof_bodyid[dofid]
    d.qfrc_bias[worldid, dofid] = wp.dot(d.cdof[worldid, dofid], cfrc[worldid, bodyid])

  leveladr, levelsize = m.body_leveladr.numpy(), m.body_levelsize.numpy()

  wp.launch(cacc_gravity, dim=[d.nworld], inputs=[m, cacc])

  for adr, size in zip(leveladr, levelsize):
    wp.launch(cacc_level, dim=(d.nworld, size), inputs=[m, d, cacc, adr])

  wp.launch(frc, dim=[d.nworld, m.nbody], inputs=[d, cfrc, cacc])

  for i in range(len(leveladr) - 1, 0, -1):
    adr, size = leveladr[i], levelsize[i]
    wp.launch(cfrc_fn, dim=[d.nworld, size], inputs=[m, cfrc, adr])

  wp.launch(qfrc_bias, dim=[d.nworld, m.nv], inputs=[m, d, cfrc])
