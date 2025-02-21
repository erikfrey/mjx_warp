# Copyright 2025 The Physics-Next Project Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import warp as wp
import mujoco
import numpy as np
import warp as wp

import mujoco

from . import support
from . import types


def put_model(mjm: mujoco.MjModel) -> types.Model:
  m = types.Model()
  m.nq = mjm.nq
  m.nv = mjm.nv
  m.na = mjm.na
  m.nu = mjm.nu
  m.nbody = mjm.nbody
  m.njnt = mjm.njnt
  m.ngeom = mjm.ngeom
  m.nsite = mjm.nsite
  m.neq = mjm.neq
  m.nmocap = mjm.nmocap
  m.nM = mjm.nM
  m.opt.gravity = wp.vec3(mjm.opt.gravity)
  m.opt.is_sparse = support.is_sparse(mjm)
  m.opt.timestep = mjm.opt.timestep
  m.opt.disableflags = mjm.opt.disableflags

  m.qpos0 = wp.array(mjm.qpos0, dtype=wp.float32, ndim=1)
  m.qpos_spring = wp.array(mjm.qpos_spring, dtype=wp.float32, ndim=1)

  # body_tree is BFS ordering of body ids
  # body_treeadr contains starting index of each body tree level
  bodies, body_depth = {}, np.zeros(mjm.nbody, dtype=int) - 1
  for i in range(mjm.nbody):
    body_depth[i] = body_depth[mjm.body_parentid[i]] + 1
    bodies.setdefault(body_depth[i], []).append(i)
  body_tree = np.concatenate([bodies[i] for i in range(len(bodies))])
  tree_off = [0] + [len(bodies[i]) for i in range(len(bodies))]
  body_treeadr = np.cumsum(tree_off)[:-1]

  m.body_tree = wp.array(body_tree, dtype=wp.int32, ndim=1)
  m.body_treeadr = wp.array(body_treeadr, dtype=wp.int32, ndim=1, device="cpu")

  qLD_update_tree = np.empty(shape=(0, 3), dtype=int)
  qLD_update_treeadr = np.empty(shape=(0,), dtype=int)
  qLD_tile = np.empty(shape=(0,), dtype=int)
  qLD_tileadr = np.empty(shape=(0,), dtype=int)
  qLD_tilesize = np.empty(shape=(0,), dtype=int)

  if support.is_sparse(mjm):
    # qLD_update_tree has dof tree ordering of qLD updates for sparse factor m
    # qLD_update_treeadr contains starting index of each dof tree level
    qLD_updates, dof_depth = {}, np.zeros(mjm.nv, dtype=int) - 1
    for k in range(mjm.nv):
      dof_depth[k] = dof_depth[mjm.dof_parentid[k]] + 1
      i = mjm.dof_parentid[k]
      Madr_ki = mjm.dof_Madr[k] + 1
      while i > -1:
        qLD_updates.setdefault(dof_depth[i], []).append((i, k, Madr_ki))
        i = mjm.dof_parentid[i]
        Madr_ki += 1

    # qLD_treeadr contains starting indicies of each level of sparse updates
    qLD_update_tree = np.concatenate([qLD_updates[i] for i in range(len(qLD_updates))])
    tree_off = [0] + [len(qLD_updates[i]) for i in range(len(qLD_updates))]
    qLD_update_treeadr = np.cumsum(tree_off)[:-1]
  else:
    # qLD_tile has the dof id of each tile in qLD for dense factor m
    # qLD_tileadr contains starting index in qLD_tile of each tile group
    # qLD_tilesize has the square tile size of each tile group
    tile_corners = [i for i in range(mjm.nv) if mjm.dof_parentid[i] == -1]
    tiles = {}
    for i in range(len(tile_corners)):
      tile_beg = tile_corners[i]
      tile_end = mjm.nv if i == len(tile_corners) - 1 else tile_corners[i + 1]
      tiles.setdefault(tile_end - tile_beg, []).append(tile_beg)
    qLD_tile = np.concatenate([tiles[sz] for sz in sorted(tiles.keys())])
    tile_off = [0] + [len(tiles[sz]) for sz in sorted(tiles.keys())]
    qLD_tileadr = np.cumsum(tile_off)[:-1]
    qLD_tilesize = np.array(sorted(tiles.keys()))

  m.qLD_update_tree = wp.array(qLD_update_tree, dtype=wp.vec3i, ndim=1)
  m.qLD_update_treeadr = wp.array(
    qLD_update_treeadr, dtype=wp.int32, ndim=1, device="cpu"
  )
  m.qLD_tile = wp.array(qLD_tile, dtype=wp.int32, ndim=1)
  m.qLD_tileadr = wp.array(qLD_tileadr, dtype=wp.int32, ndim=1, device="cpu")
  m.qLD_tilesize = wp.array(qLD_tilesize, dtype=wp.int32, ndim=1, device="cpu")
  m.body_dofadr = wp.array(mjm.body_dofadr, dtype=wp.int32, ndim=1)
  m.body_dofnum = wp.array(mjm.body_dofnum, dtype=wp.int32, ndim=1)
  m.body_jntadr = wp.array(mjm.body_jntadr, dtype=wp.int32, ndim=1)
  m.body_jntnum = wp.array(mjm.body_jntnum, dtype=wp.int32, ndim=1)
  m.body_parentid = wp.array(mjm.body_parentid, dtype=wp.int32, ndim=1)
  m.body_mocapid = wp.array(mjm.body_mocapid, dtype=wp.int32, ndim=1)
  m.body_pos = wp.array(mjm.body_pos, dtype=wp.vec3, ndim=1)
  m.body_quat = wp.array(mjm.body_quat, dtype=wp.quat, ndim=1)
  m.body_ipos = wp.array(mjm.body_ipos, dtype=wp.vec3, ndim=1)
  m.body_iquat = wp.array(mjm.body_iquat, dtype=wp.quat, ndim=1)
  m.body_rootid = wp.array(mjm.body_rootid, dtype=wp.int32, ndim=1)
  m.body_inertia = wp.array(mjm.body_inertia, dtype=wp.vec3, ndim=1)
  m.body_mass = wp.array(mjm.body_mass, dtype=wp.float32, ndim=1)
  m.body_invweight0 = wp.array(mjm.body_invweight0, dtype=wp.float32, ndim=2)
  m.jnt_bodyid = wp.array(mjm.jnt_bodyid, dtype=wp.int32, ndim=1)
  m.jnt_limited = wp.array(mjm.jnt_limited, dtype=wp.int32, ndim=1)
  m.jnt_type = wp.array(mjm.jnt_type, dtype=wp.int32, ndim=1)
  m.jnt_solref = wp.array(mjm.jnt_solref, dtype=wp.float32, ndim=2)
  m.jnt_solimp = wp.array(mjm.jnt_solimp, dtype=wp.float32, ndim=2)
  m.jnt_qposadr = wp.array(mjm.jnt_qposadr, dtype=wp.int32, ndim=1)
  m.jnt_dofadr = wp.array(mjm.jnt_dofadr, dtype=wp.int32, ndim=1)
  m.jnt_axis = wp.array(mjm.jnt_axis, dtype=wp.vec3, ndim=1)
  m.jnt_pos = wp.array(mjm.jnt_pos, dtype=wp.vec3, ndim=1)
  m.jnt_range = wp.array(mjm.jnt_range, dtype=wp.float32, ndim=2)
  m.jnt_margin = wp.array(mjm.jnt_margin, dtype=wp.float32, ndim=1)
  m.jnt_stiffness = wp.array(mjm.jnt_stiffness, dtype=wp.float32, ndim=1)
  m.geom_bodyid = wp.array(mjm.geom_bodyid, dtype=wp.int32, ndim=1)
  m.geom_pos = wp.array(mjm.geom_pos, dtype=wp.vec3, ndim=1)
  m.geom_quat = wp.array(mjm.geom_quat, dtype=wp.quat, ndim=1)
  m.site_bodyid = wp.array(mjm.site_bodyid, dtype=wp.int32, ndim=1)
  m.site_pos = wp.array(mjm.site_pos, dtype=wp.vec3, ndim=1)
  m.site_quat = wp.array(mjm.site_quat, dtype=wp.quat, ndim=1)
  m.dof_bodyid = wp.array(mjm.dof_bodyid, dtype=wp.int32, ndim=1)
  m.dof_jntid = wp.array(mjm.dof_jntid, dtype=wp.int32, ndim=1)
  m.dof_parentid = wp.array(mjm.dof_parentid, dtype=wp.int32, ndim=1)
  m.dof_Madr = wp.array(mjm.dof_Madr, dtype=wp.int32, ndim=1)
  m.dof_solref = wp.array(mjm.dof_solref, dtype=wp.float32, ndim=2)
  m.dof_solimp = wp.array(mjm.dof_solimp, dtype=wp.float32, ndim=2)
  m.dof_frictionloss = wp.array(mjm.dof_frictionloss, dtype=wp.float32, ndim=1)
  m.dof_armature = wp.array(mjm.dof_armature, dtype=wp.float32, ndim=1)
  m.dof_damping = wp.array(mjm.dof_damping, dtype=wp.float32, ndim=1)
  m.dof_invweight0 = wp.array(mjm.dof_invweight0, dtype=wp.float32, ndim=1)
  m.eq_type = wp.array(mjm.eq_type, dtype=wp.int32, ndim=1)
  m.eq_obj1id = wp.array(mjm.eq_obj1id, dtype=wp.int32, ndim=1)
  m.eq_obj2id = wp.array(mjm.eq_obj2id, dtype=wp.int32, ndim=1)
  m.eq_objtype = wp.array(mjm.eq_objtype, dtype=wp.int32, ndim=1)
  m.eq_solref = wp.array(mjm.eq_solref, dtype=wp.float32, ndim=2)
  m.eq_solimp = wp.array(mjm.eq_solimp, dtype=wp.float32, ndim=2)
  m.eq_data = wp.array(mjm.eq_data, dtype=wp.float32, ndim=2)
  m.opt.gravity = wp.vec3(mjm.opt.gravity)
  m.opt.is_sparse = support.is_sparse(mjm)
  m.opt.cone = mjm.opt.cone
  m.opt.disableflags = mjm.opt.disableflags
  m.opt.timestep = wp.float32(mjm.opt.timestep)
  m.opt.impratio = wp.float32(mjm.opt.impratio)
  # Pre-build the permanent constraint rows
  neq_connect = int(sum(mjm.eq_type == types.MJ_EQ_CONNECT))
  neq_weld = int(sum(mjm.eq_type == types.MJ_EQ_WELD))
  neq_joint = int(sum(mjm.eq_type == types.MJ_EQ_JOINT))
  nf_dof = int(sum(mjm.dof_frictionloss > 0))
  nl_ball = int(sum((mjm.jnt_type == types.MJ_JNT_BALL) * mjm.jnt_limited))
  nl_slide_hinge = int(
    sum(
      ((mjm.jnt_type == types.MJ_JNT_SLIDE) + (mjm.jnt_type == types.MJ_JNT_HINGE))
      * mjm.jnt_limited
    )
  )
  i_c = wp.zeros(6, dtype=int)
  m.efc_eq_connect_id = wp.empty(shape=(neq_connect), dtype=wp.int32)
  m.efc_eq_weld_id = wp.empty(shape=(neq_weld), dtype=wp.int32)
  m.efc_eq_joint_id = wp.empty(shape=(neq_joint), dtype=wp.int32)
  m.efc_dof_friction_id = wp.empty(shape=(nf_dof), dtype=wp.int32)
  m.efc_jnt_ball_id = wp.empty(shape=(nl_ball), dtype=wp.int32)
  m.efc_jnt_slide_hinge_id = wp.empty(shape=(nl_slide_hinge), dtype=wp.int32)
  wp.launch(
    _prebuild_efc_eq,
    dim=(mjm.eq_type.size),
    inputs=[m, i_c],
  )
  wp.launch(
    _prebuild_efc_dof_friction,
    dim=(mjm.nv),
    inputs=[m, i_c],
  )
  wp.launch(
    _prebuild_efc_jnt_limit,
    dim=(mjm.njnt),
    inputs=[m, i_c],
  )

  return m


@wp.kernel
def _prebuild_efc_eq(
  m: types.Model,
  i_c: wp.array(dtype=wp.int32),
):
  id = wp.tid()
  if m.eq_type[id] == types.MJ_EQ_CONNECT:
    irow = wp.atomic_add(i_c, 0, 1)
    m.efc_eq_connect_id[irow] = id
  elif m.eq_type[id] == types.MJ_EQ_WELD:
    irow = wp.atomic_add(i_c, 1, 1)
    m.efc_eq_weld_id[irow] = id
  elif m.eq_type[id] == types.MJ_EQ_JOINT:
    irow = wp.atomic_add(i_c, 2, 1)
    m.efc_eq_joint_id[irow] = id


@wp.kernel
def _prebuild_efc_dof_friction(
  m: types.Model,
  i_c: wp.array(dtype=wp.int32),
):
  id = wp.tid()
  if m.dof_frictionloss[id] > 0:
    irow = wp.atomic_add(i_c, 3, 1)
    m.efc_dof_friction_id[irow] = id


@wp.kernel
def _prebuild_efc_jnt_limit(
  m: types.Model,
  i_c: wp.array(dtype=wp.int32),
):
  id = wp.tid()
  if (m.jnt_type[id] == types.MJ_JNT_BALL) and m.jnt_limited[id]:
    irow = wp.atomic_add(i_c, 4, 1)
    m.efc_jnt_ball_id[irow] = id
  elif (
    m.jnt_type[id] == types.MJ_JNT_SLIDE or m.jnt_type[id] == types.MJ_JNT_HINGE
  ) and m.jnt_limited[id]:
    irow = wp.atomic_add(i_c, 5, 1)
    m.efc_jnt_slide_hinge_id[irow] = id


def make_data(mjm: mujoco.MjModel, nworld: int = 1) -> types.Data:
  d = types.Data()
  d.nworld = nworld
  d.ncon = 0
  d.nefc = 0
  d.ne = 0
  d.nf = 0
  d.nl = 0
  d.time = 0.0

  qpos0 = np.tile(mjm.qpos0, (nworld, 1))
  d.qpos = wp.array(qpos0, dtype=wp.float32, ndim=2)
  d.eq_active = wp.array((nworld, mjm.neq), dtype=wp.int32)
  d.qvel = wp.zeros((nworld, mjm.nv), dtype=wp.float32, ndim=2)
  d.qfrc_applied = wp.zeros((nworld, mjm.nv), dtype=wp.float32, ndim=2)
  d.mocap_pos = wp.zeros((nworld, mjm.nmocap), dtype=wp.vec3)
  d.mocap_quat = wp.zeros((nworld, mjm.nmocap), dtype=wp.quat)
  d.qacc = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.xanchor = wp.zeros((nworld, mjm.njnt), dtype=wp.vec3)
  d.xaxis = wp.zeros((nworld, mjm.njnt), dtype=wp.vec3)
  d.xmat = wp.zeros((nworld, mjm.nbody), dtype=wp.mat33)
  d.xpos = wp.zeros((nworld, mjm.nbody), dtype=wp.vec3)
  d.xquat = wp.zeros((nworld, mjm.nbody), dtype=wp.quat)
  d.xipos = wp.zeros((nworld, mjm.nbody), dtype=wp.vec3)
  d.ximat = wp.zeros((nworld, mjm.nbody), dtype=wp.mat33)
  d.subtree_com = wp.zeros((nworld, mjm.nbody), dtype=wp.vec3)
  d.geom_xpos = wp.zeros((nworld, mjm.ngeom), dtype=wp.vec3)
  d.geom_xmat = wp.zeros((nworld, mjm.ngeom), dtype=wp.mat33)
  d.site_xpos = wp.zeros((nworld, mjm.nsite), dtype=wp.vec3)
  d.site_xmat = wp.zeros((nworld, mjm.nsite), dtype=wp.mat33)
  d.cinert = wp.zeros((nworld, mjm.nbody), dtype=types.vec10)
  d.cdof = wp.zeros((nworld, mjm.nv), dtype=wp.spatial_vector)
  d.actuator_moment = wp.zeros((nworld, mjm.nu, mjm.nv), dtype=wp.float32)
  d.crb = wp.zeros((nworld, mjm.nbody), dtype=types.vec10)
  if support.is_sparse(mjm):
    d.qM = wp.zeros((nworld, 1, mjm.nM), dtype=wp.float32)
    d.qLD = wp.zeros((nworld, 1, mjm.nM), dtype=wp.float32)
  else:
    d.qM = wp.zeros((nworld, mjm.nv, mjm.nv), dtype=wp.float32)
    d.qLD = wp.zeros((nworld, mjm.nv, mjm.nv), dtype=wp.float32)
  d.act_dot = wp.zeros((nworld, mjm.na), dtype=wp.float32)
  d.act = wp.zeros((nworld, mjm.na), dtype=wp.float32)
  d.qLDiagInv = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.actuator_velocity = wp.zeros((nworld, mjm.nu), dtype=wp.float32)
  d.cvel = wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector)
  d.cdof_dot = wp.zeros((nworld, mjm.nv), dtype=wp.spatial_vector)
  d.qfrc_bias = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.contact = types.Contact()
  d.contact.dist = wp.zeros((nworld, d.ncon), dtype=wp.float32)
  d.contact.pos = wp.zeros((nworld, d.ncon), dtype=wp.vec3f)
  d.contact.frame = wp.zeros((nworld, d.ncon), dtype=wp.mat33f)
  d.contact.includemargin = wp.zeros((nworld, d.ncon), dtype=wp.float32)
  d.contact.friction = wp.zeros((nworld, d.ncon, 5), dtype=wp.float32)
  d.contact.solref = wp.zeros((nworld, d.ncon, types.MJ_NREF), dtype=wp.float32)
  d.contact.solreffriction = wp.zeros((nworld, d.ncon, types.MJ_NREF), dtype=wp.float32)
  d.contact.solimp = wp.zeros((nworld, d.ncon, types.MJ_NIMP), dtype=wp.float32)
  d.contact.dim = wp.zeros((nworld, d.ncon), dtype=wp.int32)
  d.contact.geom = wp.zeros((nworld, d.ncon, 2), dtype=wp.int32)
  d.contact.efc_address = wp.zeros((nworld, d.ncon), dtype=wp.int32)
  d.efc_J = wp.zeros((nworld, d.nefc, mjm.nv), dtype=wp.float32)
  d.efc_pos = wp.zeros((nworld, d.nefc), dtype=wp.float32)
  d.efc_margin = wp.zeros((nworld, d.nefc), dtype=wp.float32)
  d.efc_frictionloss = wp.zeros((nworld, d.nefc), dtype=wp.float32)
  d.efc_D = wp.zeros((nworld, d.nefc), dtype=wp.float32)
  d.efc_aref = wp.zeros((nworld, d.nefc), dtype=wp.float32)
  d.qfrc_passive = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_spring = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_damper = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_actuator = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_smooth = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_constraint = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qacc_smooth = wp.zeros((nworld, mjm.nv), dtype=wp.float32)

  # internal tmp arrays
  d.qfrc_integration = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qacc_integration = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qM_integration = wp.zeros_like(d.qM)
  d.qLD_integration = wp.zeros_like(d.qLD)
  d.qLDiagInv_integration = wp.zeros_like(d.qLDiagInv)

  return d


def put_data(mjm: mujoco.MjModel, mjd: mujoco.MjData, nworld: int = 1) -> types.Data:
  d = types.Data()
  d.nworld = nworld
  d.ncon = mjd.ncon
  d.ne = mjd.ne
  d.nf = mjd.nf
  d.nl = mjd.nl
  d.nefc = mjd.nefc
  d.time = mjd.time

  # TODO(erikfrey): would it be better to tile on the gpu?
  def tile(x):
    return np.tile(x, (nworld,) + (1,) * len(x.shape))

  if support.is_sparse(mjm):
    qM = np.expand_dims(mjd.qM, axis=0)
    qLD = np.expand_dims(mjd.qLD, axis=0)
  else:
    qM = np.zeros((mjm.nv, mjm.nv))
    mujoco.mj_fullM(mjm, qM, mjd.qM)
    qLD = np.linalg.cholesky(qM)

  # TODO(taylorhowell): sparse actuator_moment
  actuator_moment = np.zeros((mjm.nu, mjm.nv))
  mujoco.mju_sparse2dense(
    actuator_moment,
    mjd.actuator_moment,
    mjd.moment_rownnz,
    mjd.moment_rowadr,
    mjd.moment_colind,
  )

  d.qpos = wp.array(tile(mjd.qpos), dtype=wp.float32, ndim=2)
  d.eq_active = wp.array(tile(mjd.eq_active), dtype=wp.int32, ndim=2)
  d.qvel = wp.array(tile(mjd.qvel), dtype=wp.float32, ndim=2)
  d.qfrc_applied = wp.array(tile(mjd.qfrc_applied), dtype=wp.float32, ndim=2)
  d.mocap_pos = wp.array(tile(mjd.mocap_pos), dtype=wp.vec3, ndim=2)
  d.mocap_quat = wp.array(tile(mjd.mocap_quat), dtype=wp.quat, ndim=2)
  d.qacc = wp.array(tile(mjd.qacc), dtype=wp.float32, ndim=2)
  d.xanchor = wp.array(tile(mjd.xanchor), dtype=wp.vec3, ndim=2)
  d.xaxis = wp.array(tile(mjd.xaxis), dtype=wp.vec3, ndim=2)
  d.xmat = wp.array(tile(mjd.xmat), dtype=wp.mat33, ndim=2)
  d.xpos = wp.array(tile(mjd.xpos), dtype=wp.vec3, ndim=2)
  d.xquat = wp.array(tile(mjd.xquat), dtype=wp.quat, ndim=2)
  d.xipos = wp.array(tile(mjd.xipos), dtype=wp.vec3, ndim=2)
  d.ximat = wp.array(tile(mjd.ximat), dtype=wp.mat33, ndim=2)
  d.subtree_com = wp.array(tile(mjd.subtree_com), dtype=wp.vec3, ndim=2)
  d.geom_xpos = wp.array(tile(mjd.geom_xpos), dtype=wp.vec3, ndim=2)
  d.geom_xmat = wp.array(tile(mjd.geom_xmat), dtype=wp.mat33, ndim=2)
  d.site_xpos = wp.array(tile(mjd.site_xpos), dtype=wp.vec3, ndim=2)
  d.site_xmat = wp.array(tile(mjd.site_xmat), dtype=wp.mat33, ndim=2)
  d.cinert = wp.array(tile(mjd.cinert), dtype=types.vec10, ndim=2)
  d.cdof = wp.array(tile(mjd.cdof), dtype=wp.spatial_vector, ndim=2)
  d.actuator_moment = wp.array(tile(actuator_moment), dtype=wp.float32, ndim=3)
  d.crb = wp.array(tile(mjd.crb), dtype=types.vec10, ndim=2)
  d.qM = wp.array(tile(qM), dtype=wp.float32, ndim=3)
  d.qLD = wp.array(tile(qLD), dtype=wp.float32, ndim=3)
  d.qLDiagInv = wp.array(tile(mjd.qLDiagInv), dtype=wp.float32, ndim=2)
  d.actuator_velocity = wp.array(tile(mjd.actuator_velocity), dtype=wp.float32, ndim=2)
  d.cvel = wp.array(tile(mjd.cvel), dtype=wp.spatial_vector, ndim=2)
  d.cdof_dot = wp.array(tile(mjd.cdof_dot), dtype=wp.spatial_vector, ndim=2)
  d.qfrc_bias = wp.array(tile(mjd.qfrc_bias), dtype=wp.float32, ndim=2)
  d.qfrc_passive = wp.array(tile(mjd.qfrc_passive), dtype=wp.float32, ndim=2)
  d.qfrc_spring = wp.array(tile(mjd.qfrc_spring), dtype=wp.float32, ndim=2)
  d.qfrc_damper = wp.array(tile(mjd.qfrc_damper), dtype=wp.float32, ndim=2)
  d.qfrc_actuator = wp.array(tile(mjd.qfrc_actuator), dtype=wp.float32, ndim=2)
  d.qfrc_smooth = wp.array(tile(mjd.qfrc_smooth), dtype=wp.float32, ndim=2)
  d.qfrc_constraint = wp.array(tile(mjd.qfrc_constraint), dtype=wp.float32, ndim=2)
  d.qacc_smooth = wp.array(tile(mjd.qacc_smooth), dtype=wp.float32, ndim=2)
  d.act = wp.array(tile(mjd.act), dtype=wp.float32, ndim=2)
  d.act_dot = wp.array(tile(mjd.act_dot), dtype=wp.float32, ndim=2)
  d.contact.dist = wp.array(tile(mjd.contact.dist), dtype=wp.float32, ndim=2)
  d.contact.pos = wp.array(tile(mjd.contact.pos), dtype=wp.vec3f, ndim=2)
  d.contact.frame = wp.array(tile(mjd.contact.frame), dtype=wp.mat33f, ndim=2)
  d.contact.includemargin = wp.array(
    tile(mjd.contact.includemargin), dtype=wp.float32, ndim=2
  )
  d.contact.friction = wp.array(tile(mjd.contact.friction), dtype=wp.float32, ndim=3)
  d.contact.solref = wp.array(tile(mjd.contact.solref), dtype=wp.float32, ndim=3)
  d.contact.solreffriction = wp.array(
    tile(mjd.contact.solreffriction), dtype=wp.float32, ndim=3
  )
  d.contact.solimp = wp.array(tile(mjd.contact.solimp), dtype=wp.float32, ndim=3)
  d.contact.dim = wp.array(tile(mjd.contact.dim), dtype=wp.int32, ndim=2)
  d.contact.geom = wp.array(tile(mjd.contact.geom), dtype=wp.int32, ndim=3)
  d.contact.efc_address = wp.array(
    tile(mjd.contact.efc_address), dtype=wp.int32, ndim=2
  )
  d.efc_J = wp.zeros((nworld, mjd.nefc, mjm.nv), dtype=wp.float32)
  d.efc_pos = wp.zeros((nworld, mjd.nefc), dtype=wp.float32)
  d.efc_margin = wp.zeros((nworld, mjd.nefc), dtype=wp.float32)
  d.efc_frictionloss = wp.zeros((nworld, mjd.nefc), dtype=wp.float32)
  d.efc_D = wp.zeros((nworld, mjd.nefc), dtype=wp.float32)
  d.efc_aref = wp.zeros((nworld, mjd.nefc), dtype=wp.float32)

  # internal tmp arrays
  d.qfrc_integration = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qacc_integration = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qM_integration = wp.zeros_like(d.qM)
  d.qLD_integration = wp.zeros_like(d.qLD)
  d.qLDiagInv_integration = wp.zeros_like(d.qLDiagInv)

  return d
