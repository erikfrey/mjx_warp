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
from . import types


@wp.func
def _update_efc_row(
  m: types.Model,
  d: types.Data,
  worldid: wp.int32,
  irow: wp.int32,
  J: wp.array(ndim=1, dtype=wp.float32),
  pos_aref: wp.float32,
  pos_imp: wp.float32,
  invweight: wp.float32,
  solref: wp.array(ndim=1, dtype=wp.float32),
  solimp: wp.array(ndim=1, dtype=wp.float32),
  margin: wp.float32,
  refsafe: bool,
):
  # Calculate kbi
  timeconst = solref[0]
  dampratio = solref[1]
  dmin = solimp[0]
  dmax = solimp[1]
  width = solimp[2]
  mid = solimp[3]
  power = solimp[4]

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
  k = wp.select(solref[0] <= 0, k, -solref[0] / (dmax * dmax))
  b = wp.select(solref[1] <= 0, b, -solref[1] / dmax)

  imp_x = wp.abs(pos_imp) / width
  imp_a = (1.0 / wp.pow(mid, power - 1.0)) * wp.pow(imp_x, power)
  imp_b = 1.0 - (1.0 / wp.pow(1.0 - mid, power - 1.0)) * wp.pow(1.0 - imp_x, power)
  imp_y = wp.select(imp_x < mid, imp_b, imp_a)
  imp = dmin + imp_y * (dmax - dmin)
  imp = wp.clamp(imp, dmin, dmax)
  imp = wp.select(imp_x > 1.0, imp, dmax)

  # Update constraints
  r = wp.max(invweight * (1.0 - imp) / imp, types.MJ_MINVAL)
  aref = float(0.0)
  for i in range(m.nv):
    aref += -b * (J[i] * d.qvel[worldid, i])
    d.efc_J[worldid, irow, i] = J[i]
  aref -= k * imp * pos_aref
  d.efc_D[worldid, irow] = 1.0 / r
  d.efc_aref[worldid, irow] = aref
  d.efc_pos[worldid, irow] = pos_aref + margin
  d.efc_margin[worldid, irow] = margin

  # Update the number of constraints
  wp.atomic_add(d.nefc, 0, 1)


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

  return jacp


@wp.kernel
def _efc_limit_slide_hinge(
  m: types.Model,
  d: types.Data,
  i_c: wp.array(ndim=1, dtype=wp.int32),
  jnt_id: wp.array(ndim=1, dtype=wp.int32),
  J: wp.array(ndim=2, dtype=wp.float32),
  refsafe: bool,
):
  worldid, id = wp.tid()
  n_id = jnt_id[id]

  qpos = d.qpos[worldid, m.jnt_qposadr[n_id]]
  dist_min, dist_max = qpos - m.jnt_range[n_id][0], m.jnt_range[n_id][1] - qpos
  pos = wp.min(dist_min, dist_max) - m.jnt_margin[n_id]
  active = pos < 0
  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    J[irow, m.jnt_dofadr[n_id]] = float(dist_min < dist_max) * 2.0 - 1.0

    _update_efc_row(
      m,
      d,
      worldid,
      irow,
      J[irow],
      pos,
      pos,
      m.dof_invweight0[m.jnt_dofadr[n_id]],
      m.jnt_solref[n_id],
      m.jnt_solimp[n_id],
      m.jnt_margin[n_id],
      refsafe,
    )


@wp.kernel
def _efc_contact_pyramidal(
  m: types.Model,
  d: types.Data,
  worldid: wp.array(ndim=1, dtype=wp.int32),
  i_c: wp.array(ndim=1, dtype=wp.int32),
  con_id: wp.array(ndim=1, dtype=wp.int32),
  con_dim_id: wp.array(ndim=1, dtype=wp.int32),
  J: wp.array(ndim=2, dtype=wp.float32),
  refsafe: bool,
):
  id = wp.tid()
  if id < i_c[1]:
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
      + d.contact.friction[w_id, n_id, 0]
      * d.contact.friction[w_id, n_id, 0]
      * invweight
    )

    active = pos < 0
    if active:
      irow = wp.atomic_add(i_c, 0, 1)
      invweight = (
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
          jac1p = _jac(m, d, w_id, d.contact.pos[w_id, n_id], xyz, body1, i)
          jac2p = _jac(m, d, w_id, d.contact.pos[w_id, n_id], xyz, body2, i)
          diff_0 += d.contact.frame[w_id, n_id][0, xyz] * (jac2p - jac1p)
          diff_i += d.contact.frame[w_id, n_id][con_dim, xyz] * (jac2p - jac1p)
        if id % 2 == 0:
          J[irow, i] = diff_0 + diff_i * d.contact.friction[w_id, n_id, con_dim - 1]
        else:
          J[irow, i] = diff_0 - diff_i * d.contact.friction[w_id, n_id, con_dim - 1]

      _update_efc_row(
        m,
        d,
        w_id,
        irow,
        J[irow],
        pos,
        pos,
        invweight,
        d.contact.solref[w_id, n_id],
        d.contact.solimp[w_id, n_id],
        d.contact.includemargin[w_id, n_id],
        refsafe,
      )


@wp.kernel
def _allocate_efc_contact_pyramidal(
  d: types.Data,
  i_c: wp.array(dtype=wp.int32),
  worldid_con: wp.array(dtype=wp.int32),
  con_id: wp.array(dtype=wp.int32),
  con_dim_id: wp.array(dtype=wp.int32),
):
  w_id, i_con = wp.tid()
  condim = 3

  if d.contact.dim[w_id, i_con] == condim:
    irow = wp.atomic_add(i_c, 1, 2 * (condim - 1))
    for j in range(condim - 1):
      con_id[irow + 2 * j] = i_con
      con_id[irow + 2 * j + 1] = i_con
      con_dim_id[irow + 2 * j] = j + 1
      con_dim_id[irow + 2 * j + 1] = j + 1
      worldid_con[irow + 2 * j] = w_id
      worldid_con[irow + 2 * j + 1] = w_id


def make_constraint(m: types.Model, d: types.Data):
  """Creates constraint jacobians and other supporting data."""

  i_c = wp.zeros(2, dtype=int)
  if not (m.opt.disableflags & types.DisableBit.CONSTRAINT.value):
    # Prepare the contact constraints using conservative values
    n_con = d.ncon * d.nworld * 6
    worldid_con = wp.empty(n_con, dtype=wp.int32)
    con_id = wp.empty(n_con, dtype=wp.int32)
    com_dim_id = wp.empty(n_con, dtype=wp.int32)
    wp.launch(
      _allocate_efc_contact_pyramidal,
      dim=(d.nworld, d.ncon),
      inputs=[d, i_c],
      outputs=[worldid_con, con_id, com_dim_id],
    )

    refsafe = not m.opt.disableflags & types.DisableBit.REFSAFE.value

    if not (m.opt.disableflags & types.DisableBit.LIMIT.value) and (
      m.efc_jnt_slide_hinge_id.size != 0
    ):
      temp_jac = wp.zeros((m.efc_jnt_slide_hinge_id.size, m.nv), dtype=wp.float32)
      wp.launch(
        _efc_limit_slide_hinge,
        dim=(d.nworld, m.efc_jnt_slide_hinge_id.size),
        inputs=[m, d, i_c, m.efc_jnt_slide_hinge_id, temp_jac, refsafe],
      )

    if m.opt.cone == types.ConeType.PYRAMIDAL.value:
      temp_jac = wp.zeros((n_con, m.nv), dtype=wp.float32)
      wp.launch(
        _efc_contact_pyramidal,
        dim=n_con,
        inputs=[m, d, worldid_con, i_c, con_id, com_dim_id, temp_jac, refsafe],
      )

  return d
