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
from . import math
from . import types
from typing import Any


@wp.struct
class _Efc:
  worldid: wp.array(ndim=1, dtype=wp.int32)
  J: wp.array(ndim=2, dtype=wp.float32)
  pos_aref: wp.array(ndim=2, dtype=wp.float32)
  pos_imp: wp.array(ndim=2, dtype=wp.float32)
  invweight: wp.array(ndim=2, dtype=wp.float32)
  solref: wp.array(ndim=2, dtype=wp.float32)
  solimp: wp.array(ndim=2, dtype=wp.float32)
  margin: wp.array(ndim=2, dtype=wp.float32)
  frictionloss: wp.array(ndim=2, dtype=wp.float32)


@wp.kernel
def _update_contact_data(m: types.Model, d: types.Data, efcs: _Efc, refsafe: bool):
  id = wp.tid()
  worldid = efcs.worldid[id]

  # Calculate kbi
  timeconst = efcs.solref[id, 0]
  dampratio = efcs.solref[id, 1]
  dmin = efcs.solimp[id, 0]
  dmax = efcs.solimp[id, 1]
  width = efcs.solimp[id, 2]
  mid = efcs.solimp[id, 3]
  power = efcs.solimp[id, 4]

  if refsafe:
    timeconst = wp.max(timeconst, 2.0 * m.opt.timestep)

  dmin = wp.clamp(dmin, types.MJ_MINIMP, types.MJ_MAXIMP)
  dmax = wp.clamp(dmax, types.MJ_MINIMP, types.MJ_MAXIMP)
  width = wp.max(types.MJ_MINVAL, width)
  mid = wp.clamp(mid, types.MJ_MINIMP, types.MJ_MAXIMP)
  power = wp.max(1.0, power)

  # See https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
  k = 1.0 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)
  b = 2.0 / (dmax * timeconst)
  # TODO(robotics-simulation): check various solparam settings in model gen test
  k = wp.select(efcs.solref[id, 0] <= 0, k, -efcs.solref[id, 0] / (dmax * dmax))
  b = wp.select(efcs.solref[id, 1] <= 0, b, -efcs.solref[id, 1] / dmax)

  imp_x = wp.abs(efcs.pos_imp[id, 0]) / width
  imp_a = (1.0 / wp.pow(mid, power - 1.0)) * wp.pow(imp_x, power)
  imp_b = 1.0 - (1.0 / wp.pow(1.0 - mid, power - 1.0)) * wp.pow(1.0 - imp_x, power)
  imp_y = wp.select(imp_x < mid, imp_b, imp_a)
  imp = dmin + imp_y * (dmax - dmin)
  imp = wp.clamp(imp, dmin, dmax)
  imp = wp.select(imp_x > 1.0, imp, dmax)

  # Update constraints
  r = wp.max(efcs.invweight[id, 0] * (1.0 - imp) / imp, types.MJ_MINVAL)
  aref = float(0.0)
  for i in range(m.nv):
    aref += -b * (efcs.J[id, i] * d.qvel[worldid, i])
    d.efc_J[worldid, id, i] = efcs.J[id, i]
  aref -= k * imp * efcs.pos_aref[id, 0]
  d.efc_D[worldid, id] = 1.0 / r
  d.efc_aref[worldid, id] = aref
  d.efc_pos[worldid, id] = efcs.pos_aref[id, 0] + efcs.margin[id, 0]
  d.efc_margin[worldid, id] = efcs.margin[id, 0]
  d.efc_frictionloss[worldid, id] = efcs.frictionloss[id, 0]


@wp.func
def _jac(
  m: types.Model,
  d: types.Data,
  worldid: wp.int32,
  point: wp.vec3,
  xyz: wp.int32,
  bodyid: wp.int32,
  dofid: wp.int32,
):
  dof_bodyid = m.dof_bodyid[dofid]
  in_tree = int(dof_bodyid == 0)
  parentid = bodyid
  while parentid != 0:
    if parentid == dof_bodyid:
      in_tree = 1
      break
    parentid = m.body_parentid[parentid]

  offset = point - wp.vec3(d.subtree_com[worldid, m.body_rootid[bodyid]])

  temp_jac = wp.vec3(0.0)
  temp_jac = wp.spatial_bottom(d.cdof[worldid, dofid]) + wp.cross(
    wp.spatial_top(d.cdof[worldid, dofid]), offset
  )
  jacp = temp_jac[xyz] * float(in_tree)
  jacr = d.cdof[worldid, dofid][xyz] * float(in_tree)

  return jacp, jacr


@wp.kernel
def _efc_limit_slide_hinge(
  m: types.Model,
  d: types.Data,
  i_c: wp.array(dtype=wp.int32),
  jnt_id: wp.array(dtype=wp.int32),
  limit_slide_hinge: _Efc,
):
  worldid, id = wp.tid()
  n_id = jnt_id[id]

  qpos = d.qpos[worldid, m.jnt_qposadr[n_id]]
  dist_min, dist_max = qpos - m.jnt_range[n_id][0], m.jnt_range[n_id][1] - qpos
  pos = wp.min(dist_min, dist_max) - m.jnt_margin[n_id]
  active = pos < 0
  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    limit_slide_hinge.worldid[irow] = worldid
    limit_slide_hinge.pos_imp[irow, 0] = pos
    for i in range(types.MJ_NREF):
      limit_slide_hinge.solref[irow, i] = m.jnt_solref[n_id, i]
    for i in range(types.MJ_NIMP):
      limit_slide_hinge.solimp[irow, i] = m.jnt_solimp[n_id, i]
    limit_slide_hinge.margin[irow, 0] = m.jnt_margin[n_id]
    limit_slide_hinge.invweight[irow, 0] = m.dof_invweight0[m.jnt_dofadr[n_id]]
    limit_slide_hinge.J[irow, m.jnt_dofadr[n_id]] = (
      float(dist_min < dist_max) * 2.0 - 1.0
    )
    limit_slide_hinge.pos_aref[irow, 0] = pos


@wp.kernel
def _efc_contact_pyramidal(
  m: types.Model,
  d: types.Data,
  worldid: wp.array(dtype=wp.int32),
  i_c: wp.array(dtype=wp.int32),
  con_id: wp.array(dtype=wp.int32),
  con_dim_id: wp.array(dtype=wp.int32),
  contact_pyramidal: _Efc,
):
  id = wp.tid()
  n_id = con_id[id]
  w_id = worldid[id]
  con_dim = con_dim_id[id]

  pos = d.contact.dist[w_id, n_id] - d.contact.includemargin[w_id, n_id]

  body1 = m.geom_bodyid[d.contact.geom[w_id, n_id, 0]]
  body2 = m.geom_bodyid[d.contact.geom[w_id, n_id, 1]]

  # pyramidal has common invweight across all edges
  invweight = m.body_invweight0[body1, 0] + m.body_invweight0[body2, 0]
  invweight = (
    invweight
    + d.contact.friction[w_id, n_id, 0] * d.contact.friction[w_id, n_id, 0] * invweight
  )

  active = pos < 0
  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    contact_pyramidal.worldid[irow] = w_id
    contact_pyramidal.pos_imp[irow, 0] = pos
    for i in range(types.MJ_NREF):
      contact_pyramidal.solref[irow, i] = d.contact.solref[w_id, n_id, i]
    for i in range(types.MJ_NIMP):
      contact_pyramidal.solimp[irow, i] = d.contact.solimp[w_id, n_id, i]
    contact_pyramidal.margin[irow, 0] = d.contact.includemargin[w_id, n_id]
    contact_pyramidal.invweight[irow, 0] = (
      invweight
      * 2.0
      * d.contact.friction[w_id, n_id, 0]
      * d.contact.friction[w_id, n_id, 0]
      / m.opt.impratio
    )
    for i in range(m.nv):
      diff_0 = float(0.0)
      diff_i = float(0.0)
      for xyz in range(3):
        jac1p, jac1r = _jac(m, d, w_id, d.contact.pos[w_id, n_id], xyz, body1, i)
        jac2p, jac2r = _jac(m, d, w_id, d.contact.pos[w_id, n_id], xyz, body2, i)
        diff_0 += d.contact.frame[w_id, n_id][0, xyz] * (jac2p - jac1p)
        if con_dim < 3:
          diff_i += d.contact.frame[w_id, n_id][con_dim, xyz] * (jac2p - jac1p)
        else:
          diff_i += d.contact.frame[w_id, n_id][con_dim - 3, xyz] * (jac2r - jac1r)
      if id % 2 == 0:
        contact_pyramidal.J[irow, i] = (
          diff_0 + diff_i * d.contact.friction[w_id, n_id, con_dim - 1]
        )
      else:
        contact_pyramidal.J[irow, i] = (
          diff_0 - diff_i * d.contact.friction[w_id, n_id, con_dim - 1]
        )
    contact_pyramidal.pos_aref[irow, 0] = pos


@wp.kernel
def _allocate_efc_contact_pyramidal_elliptic(
  m: types.Model,
  d: types.Data,
  i_c: wp.array(dtype=wp.int32),
  worldid_con: wp.array(dtype=wp.int32),
  con_id: wp.array(dtype=wp.int32),
  con_dim_id: wp.array(dtype=wp.int32),
):
  id, w_id, i_con = wp.tid()
  condim = wp.vec3i(3, 4, 6)[id]

  if d.contact.dim[w_id, i_con] == condim:
    if m.opt.cone == types.MJ_CONE_PYRAMIDAL:
      irow = wp.atomic_add(i_c, 2, 2 * (condim - 1))
      for j in range(condim - 1):
        con_id[irow + 2 * j] = i_con
        con_id[irow + 2 * j + 1] = i_con
        con_dim_id[irow + 2 * j] = j + 1
        con_dim_id[irow + 2 * j + 1] = j + 1
        worldid_con[irow + 2 * j] = w_id
        worldid_con[irow + 2 * j + 1] = w_id


def make_constraint(m: types.Model, d: types.Data):
  """Creates constraint jacobians and other supporting data."""

  i_c = wp.zeros(3, dtype=int)
  if not (m.opt.disableflags & types.MJ_DSBL_CONSTRAINT):
    # Prepare the contact constraints using conservative values
    n_con = d.ncon * d.nworld * 12
    worldid_con = wp.empty(n_con, dtype=wp.int32)
    con_id = wp.empty(n_con, dtype=wp.int32)
    com_dim_id = wp.empty(n_con, dtype=wp.int32)
    wp.launch(
      _allocate_efc_contact_pyramidal_elliptic,
      dim=(3, d.nworld, d.ncon),
      inputs=[m, d, i_c],
      outputs=[worldid_con, con_id, com_dim_id],
    )

    # Allocate the constraint rows
    efcs = _Efc()
    efcs.worldid = wp.zeros(shape=(d.nefc), dtype=wp.int32)
    efcs.J = wp.zeros((d.nefc, m.nv), dtype=wp.float32)
    efcs.pos_aref = wp.zeros(shape=(d.nefc), dtype=wp.float32)
    efcs.pos_imp = wp.zeros(shape=(d.nefc), dtype=wp.float32)
    efcs.invweight = wp.zeros(shape=(d.nefc), dtype=wp.float32)
    efcs.solref = wp.zeros(shape=(d.nefc, types.MJ_NREF), dtype=wp.float32)
    efcs.solimp = wp.zeros(shape=(d.nefc, types.MJ_NIMP), dtype=wp.float32)
    efcs.margin = wp.zeros(shape=(d.nefc), dtype=wp.float32)
    efcs.frictionloss = wp.zeros(shape=(d.nefc), dtype=wp.float32)

    if not (m.opt.disableflags & types.MJ_DSBL_LIMIT) and (
      m.efc_jnt_slide_hinge_id.size != 0
    ):
      wp.launch(
        _efc_limit_slide_hinge,
        dim=(d.nworld, m.efc_jnt_slide_hinge_id.size),
        inputs=[m, d, i_c, m.efc_jnt_slide_hinge_id],
        outputs=[efcs],
      )

    con_size = int(i_c.numpy()[2])
    if con_size != 0:
      if m.opt.cone == types.MJ_CONE_PYRAMIDAL:
        wp.launch(
          _efc_contact_pyramidal,
          dim=con_size,
          inputs=[m, d, worldid_con, i_c, con_id, com_dim_id],
          outputs=[efcs],
        )

  i_c_np = int(i_c.numpy()[0])
  if i_c_np == 0:
    d.efc_J = wp.empty((0, m.nv))
    d.efc_D = wp.empty(0)
    d.efc_aref = wp.empty(0)
    d.efc_frictionloss = wp.empty(0)
    d.efc_pos = wp.empty(0)
    d.efc_margin = wp.empty(0)
    return d

  refsafe = not m.opt.disableflags & types.MJ_DSBL_REFSAFE
  wp.launch(_update_contact_data, dim=i_c_np, inputs=[m, d, efcs, refsafe])

  return d
