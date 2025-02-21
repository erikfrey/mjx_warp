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
def _efc_equality_connect(
  m: types.Model,
  d: types.Data,
  i_c: wp.array(dtype=wp.int32),
  eq_id: wp.array(dtype=wp.int32),
  equality_connect: _Efc,
):
  worldid, id = wp.tid()
  c_id = id // 3
  xyz_id = id % 3
  n_id = eq_id[c_id]

  active = d.eq_active[worldid, n_id]
  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    equality_connect.worldid[irow] = worldid
    is_site = int(m.eq_objtype[n_id] == types.MJ_OBJ_SITE)
    obj1id = m.eq_obj1id[n_id]
    obj2id = m.eq_obj2id[n_id]
    data = m.eq_data[n_id]
    for i in range(types.MJ_NREF):
      equality_connect.solref[irow, i] = m.eq_solref[n_id, i]
    for i in range(types.MJ_NIMP):
      equality_connect.solimp[irow, i] = m.eq_solimp[n_id, i]

    if is_site:
      body1id = m.site_bodyid[obj1id]
      body2id = m.site_bodyid[obj2id]
    else:
      body1id = m.eq_obj1id[n_id]
      body2id = m.eq_obj2id[n_id]

    anchor1 = wp.vec3(data[0], data[1], data[2])
    anchor2 = wp.vec3(data[3], data[4], data[5])

    pos1 = d.xmat[worldid, body1id] * anchor1 + d.xpos[worldid, body1id]
    pos2 = d.xmat[worldid, body2id] * anchor2 + d.xpos[worldid, body2id]

    if is_site:
      pos1 = d.site_xpos[worldid, obj1id]
      pos2 = d.site_xpos[worldid, obj2id]

    # error is difference in global positions
    pos = wp.vec3(0.0)
    for i in range(3):
      pos[i] = pos1[i] - pos2[i]
    equality_connect.pos_aref[irow, 0] = pos[xyz_id]
    equality_connect.pos_imp[irow, 0] = math.norm_l2(pos)

    # compute Jacobian difference (opposite of contact: 0 - 1)
    for i in range(m.nv):
      jacp1, _ = _jac(m, d, worldid, pos1, xyz_id, body1id, i)
      jacp2, _ = _jac(m, d, worldid, pos2, xyz_id, body2id, i)
      equality_connect.J[irow, i] = jacp1 - jacp2

    equality_connect.invweight[irow, 0] = (
      m.body_invweight0[body1id, 0] + m.body_invweight0[body2id, 0]
    )


@wp.kernel
def _efc_equality_weld(
  m: types.Model,
  d: types.Data,
  i_c: wp.array(dtype=wp.int32),
  eq_id: wp.array(dtype=wp.int32),
  equality_weld: _Efc,
):
  worldid, id = wp.tid()
  c_id = id // 6
  spatial_id = id % 6
  xyz_id = id % 3
  n_id = eq_id[c_id]

  active = d.eq_active[worldid, n_id]
  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    equality_weld.worldid[irow] = worldid
    is_site = int(m.eq_objtype[n_id] == types.MJ_OBJ_SITE)
    obj1id = m.eq_obj1id[n_id]
    obj2id = m.eq_obj2id[n_id]
    body1id = m.eq_obj1id[n_id]
    body2id = m.eq_obj2id[n_id]
    data = m.eq_data[n_id]
    for i in range(types.MJ_NREF):
      equality_weld.solref[irow, i] = m.eq_solref[n_id, i]
    for i in range(types.MJ_NIMP):
      equality_weld.solimp[irow, i] = m.eq_solimp[n_id, i]

    if is_site:
      body1id = m.site_bodyid[obj1id]
      body2id = m.site_bodyid[obj2id]
    else:
      body1id = m.eq_obj1id[n_id]
      body2id = m.eq_obj2id[n_id]

    anchor1 = wp.vec3(data[0], data[1], data[2])
    anchor2 = wp.vec3(data[3], data[4], data[5])
    relpose = wp.quat(data[6], data[7], data[8], data[9])
    torquescale = data[10]

    # error is difference in global position and orientation
    pos1 = d.xmat[worldid, body1id] * anchor2 + d.xpos[worldid, body1id]
    pos2 = d.xmat[worldid, body2id] * anchor1 + d.xpos[worldid, body2id]

    if is_site:
      pos1 = d.site_xpos[worldid, obj1id]
      pos2 = d.site_xpos[worldid, obj2id]

    # compute orientation error: neg(q1) * q0 * relpose (axis components only)
    quat = math.mul_quat(d.xquat[worldid, body1id], relpose)
    quat1 = math.quat_inv(d.xquat[worldid, body2id])

    if is_site:
      quat = math.mul_quat(d.xquat[worldid, body1id], m.site_quat[obj1id])
      quat1 = math.quat_inv(
        math.mul_quat(d.xquat[worldid, body2id], m.site_quat[obj2id])
      )

    mulquat = math.mul_quat(quat1, quat)
    crot = wp.vec3(mulquat[1], mulquat[2], mulquat[3])  # copy axis components

    pos_imp = 0.0
    for i in range(3):
      cpos = pos1[i] - pos2[i]
      pos_imp += cpos * cpos + (crot[i] * torquescale) ** 2.0
    equality_weld.pos_imp[irow, 0] = wp.sqrt(pos_imp)

    # compute Jacobian difference (opposite of contact: 0 - 1)
    for i in range(m.nv):
      if spatial_id < 3:
        jacp1, _ = _jac(m, d, worldid, pos1, xyz_id, body1id, i)
        jacp2, _ = _jac(m, d, worldid, pos2, xyz_id, body2id, i)
        jacdifp = jacp1 - jacp2
        equality_weld.J[irow, i] = jacdifp
      else:
        jacdifr = wp.vec3(0.0)
        for xyz in range(3):
          _, jacr1 = _jac(m, d, worldid, pos1, xyz, body1id, i)
          _, jacr2 = _jac(m, d, worldid, pos2, xyz, body2id, i)
          jacdifr[xyz] = (jacr1 - jacr2) * torquescale
        # correct rotation Jacobian: 0.5 * neg(q1) * (jac0-jac1) * q0 * relpose
        temp_quat = wp.quat(
          -quat1[1] * jacdifr[0] - quat1[2] * jacdifr[1] - quat1[3] * jacdifr[2],
          quat1[0] * jacdifr[0] + quat1[2] * jacdifr[2] - quat1[3] * jacdifr[1],
          quat1[0] * jacdifr[1] + quat1[3] * jacdifr[0] - quat1[1] * jacdifr[2],
          quat1[0] * jacdifr[2] + quat1[1] * jacdifr[1] - quat1[2] * jacdifr[0],
        )
        equality_weld.J[irow, i] = 0.5 * math.mul_quat(temp_quat, quat)[1 + xyz_id]

    if spatial_id < 3:
      equality_weld.pos_aref[irow, 0] = pos1[xyz_id] - pos2[xyz_id]
      equality_weld.invweight[irow, 0] = (
        m.body_invweight0[body1id, 0] + m.body_invweight0[body2id, 0]
      )
    else:
      equality_weld.pos_aref[irow, 0] = crot[xyz_id] * torquescale
      equality_weld.invweight[irow, 0] = (
        m.body_invweight0[body1id, 1] + m.body_invweight0[body2id, 1]
      )


@wp.kernel
def _efc_equality_joint(
  m: types.Model,
  d: types.Data,
  i_c: wp.array(dtype=wp.int32),
  eq_id: wp.array(dtype=wp.int32),
  equality_joint: _Efc,
):
  worldid, id = wp.tid()
  n_id = eq_id[id]

  active = d.eq_active[worldid, n_id]
  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    equality_joint.worldid[irow] = worldid
    obj1id = wp.int32(m.eq_obj1id[n_id])
    obj2id = wp.int32(m.eq_obj2id[n_id])
    dofadr1, dofadr2 = m.jnt_dofadr[obj1id], m.jnt_dofadr[obj2id]
    qposadr1, qposadr2 = m.jnt_qposadr[obj1id], m.jnt_qposadr[obj2id]
    data = m.eq_data[n_id]
    for i in range(types.MJ_NREF):
      equality_joint.solref[irow, i] = m.eq_solref[n_id, i]
    for i in range(types.MJ_NIMP):
      equality_joint.solimp[irow, i] = m.eq_solimp[n_id, i]

    pos1, pos2 = d.qpos[worldid, qposadr1], d.qpos[worldid, qposadr2]
    ref1, ref2 = m.qpos0[qposadr1], m.qpos0[qposadr2]
    dif = (pos2 - ref2) * float(obj2id > -1)
    pos = pos1 - ref1
    for i in range(5):
      pos -= data[i] * wp.pow(dif, float(i))
    equality_joint.pos_aref[irow, 0] = pos
    equality_joint.pos_imp[irow, 0] = pos

    deriv = float(0.0)
    for i in range(4):
      deriv += data[i + 1] * wp.pow(dif, float(i)) * float(i + 1) * float(obj2id > -1)
    equality_joint.J[irow, dofadr2] = -deriv
    equality_joint.J[irow, dofadr1] = 1.0

    invweight = m.dof_invweight0[dofadr1]
    invweight += m.dof_invweight0[dofadr2] * float(obj2id > -1)
    equality_joint.invweight[irow, 0] = invweight


@wp.kernel
def _efc_dof_friction(
  m: types.Model,
  i_c: wp.array(dtype=wp.int32),
  dof_id: wp.array(dtype=wp.int32),
  dof_friction: _Efc,
):
  worldid, id = wp.tid()
  n_id = dof_id[id]
  irow = wp.atomic_add(i_c, 0, 1)

  dof_friction.worldid[irow] = worldid
  dof_friction.frictionloss[irow, 0] = m.dof_frictionloss[n_id]
  dof_friction.invweight[irow, 0] = m.dof_invweight0[n_id]
  for i in range(types.MJ_NREF):
    dof_friction.solref[irow, i] = m.dof_solref[n_id, i]
  for i in range(types.MJ_NIMP):
    dof_friction.solimp[irow, i] = m.dof_solimp[n_id, i]
  dof_friction.J[irow, n_id] = 1.0


@wp.kernel
def _efc_limit_ball(
  m: types.Model,
  d: types.Data,
  i_c: wp.array(dtype=wp.int32),
  jnt_id: wp.array(dtype=wp.int32),
  limit_ball: _Efc,
):
  worldid, id = wp.tid()
  n_id = jnt_id[id]

  jntquat = wp.quat(
    d.qpos[worldid, m.jnt_qposadr[n_id]],
    d.qpos[worldid, m.jnt_qposadr[n_id] + 1],
    d.qpos[worldid, m.jnt_qposadr[n_id] + 2],
    d.qpos[worldid, m.jnt_qposadr[n_id] + 3],
  )
  axis, angle = math.quat_to_axis_angle(jntquat)
  # ball rotation angle is always positive
  angle = wp.norm_l2(axis * angle)
  axis = wp.normalize(axis * angle)
  pos = wp.max(m.jnt_range[n_id, 0], m.jnt_range[n_id, 1]) - angle - m.jnt_margin[n_id]

  active = pos < 0
  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    limit_ball.worldid[irow] = worldid
    for i in range(types.MJ_NREF):
      limit_ball.solref[irow, i] = m.jnt_solref[n_id, i]
    for i in range(types.MJ_NIMP):
      limit_ball.solimp[irow, i] = m.jnt_solimp[n_id, i]
    limit_ball.margin[irow, 0] = m.jnt_margin[n_id]
    limit_ball.invweight[irow, 0] = m.dof_invweight0[m.jnt_dofadr[n_id]]
    limit_ball.pos_imp[irow, 0] = pos
    for i in range(3):
      limit_ball.J[irow, m.jnt_dofadr[n_id] + i] = -axis[i]
    limit_ball.pos_aref[irow, 0] = pos


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
def _efc_contact_frictionless(
  m: types.Model,
  d: types.Data,
  worldid: wp.array(dtype=wp.int32),
  i_c: wp.array(dtype=wp.int32),
  con_id: wp.array(dtype=wp.int32),
  contact_frictionless: _Efc,
):
  id = wp.tid()
  n_id = con_id[id]
  w_id = worldid[id]

  body1 = m.geom_bodyid[d.contact.geom[w_id, n_id, 0]]
  body2 = m.geom_bodyid[d.contact.geom[w_id, n_id, 1]]
  pos = d.contact.dist[w_id, n_id] - d.contact.includemargin[w_id, n_id]

  active = pos < 0
  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    contact_frictionless.worldid[irow] = w_id
    for i in range(types.MJ_NREF):
      contact_frictionless.solref[irow, i] = d.contact.solref[w_id, n_id, i]
    for i in range(types.MJ_NIMP):
      contact_frictionless.solimp[irow, i] = d.contact.solimp[w_id, n_id, i]
    contact_frictionless.margin[irow, 0] = d.contact.includemargin[w_id, n_id]
    contact_frictionless.invweight[irow, 0] = (
      m.body_invweight0[body1, 0] + m.body_invweight0[body2, 0]
    )
    contact_frictionless.pos_imp[irow, 0] = pos
    for i in range(m.nv):
      for xyz in range(3):
        jacp1, _ = _jac(m, d, w_id, d.contact.pos[w_id, n_id], xyz, body1, i)
        jacp2, _ = _jac(m, d, w_id, d.contact.pos[w_id, n_id], xyz, body2, i)
        contact_frictionless.J[irow, i] += d.contact.frame[w_id, n_id][0, xyz] * (
          jacp2 - jacp1
        )
    contact_frictionless.pos_aref[irow, 0] = pos


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
def _efc_contact_elliptic(
  m: types.Model,
  d: types.Data,
  worldid: wp.array(dtype=wp.int32),
  i_c: wp.array(dtype=wp.int32),
  con_id: wp.array(dtype=wp.int32),
  con_dim_id: wp.array(dtype=wp.int32),
  contact_elliptic: _Efc,
):
  id = wp.tid()
  n_id = con_id[id]
  w_id = worldid[id]
  con_dim = con_dim_id[id]

  pos = d.contact.dist[w_id, n_id] - d.contact.includemargin[w_id, n_id]

  active = pos < 0
  obj1id = m.geom_bodyid[d.contact.geom[w_id, n_id, 0]]
  obj2id = m.geom_bodyid[d.contact.geom[w_id, n_id, 1]]
  invweight = m.body_invweight0[obj1id, 0] + m.body_invweight0[obj2id, 0]

  if active:
    irow = wp.atomic_add(i_c, 0, 1)
    contact_elliptic.worldid[irow] = w_id
    for i in range(types.MJ_NIMP):
      contact_elliptic.solimp[irow, i] = d.contact.solimp[w_id, n_id, i]
    contact_elliptic.margin[irow, 0] = d.contact.includemargin[w_id, n_id]
    contact_elliptic.pos_imp[irow, 0] = pos
    if con_dim == 0:
      contact_elliptic.invweight[irow, 0] = invweight
      for i in range(types.MJ_NREF):
        contact_elliptic.solref[irow, i] = d.contact.solref[w_id, n_id, i]
    else:
      invweight_factor = invweight / m.opt.impratio
      contact_elliptic.invweight[irow, 0] = invweight_factor * (
        d.contact.friction[w_id, n_id, 0]
        * d.contact.friction[w_id, n_id, 0]
        / (
          d.contact.friction[w_id, n_id, con_dim - 1]
          * d.contact.friction[w_id, n_id, con_dim - 1]
        )
      )
      use_solreffriction = False
      for i in range(types.MJ_NREF):
        if d.contact.solreffriction[w_id, n_id, i] != 0.0:
          use_solreffriction = True
      if use_solreffriction:
        for i in range(types.MJ_NREF):
          contact_elliptic.solref[irow, i] = d.contact.solreffriction[w_id, n_id, i]
      else:
        contact_elliptic.solref[irow, i] = d.contact.solref[w_id, n_id, i]
    for i in range(m.nv):
      for xyz in range(3):
        jac1p, jac1r = _jac(m, d, w_id, d.contact.pos[w_id, n_id], xyz, obj1id, i)
        jac2p, jac2r = _jac(m, d, w_id, d.contact.pos[w_id, n_id], xyz, obj2id, i)
        if con_dim < 3:
          contact_elliptic.J[irow, i] += d.contact.frame[w_id, n_id][con_dim, xyz] * (
            jac2p - jac1p
          )
        else:
          contact_elliptic.J[irow, i] += d.contact.frame[w_id, n_id][
            con_dim - 3, xyz
          ] * (jac2r - jac1r)

    if con_dim == 0:
      contact_elliptic.pos_aref[irow, 0] = pos


@wp.kernel
def _allocate_efc_contact_frictionless(
  d: types.Data,
  i_c: wp.array(dtype=wp.int32),
  worldid_frictionless: wp.array(dtype=wp.int32),
  con_frictionless_id: wp.array(dtype=wp.int32),
):
  w_id, i_con = wp.tid()

  if d.contact.dim[w_id, i_con] == 1:
    irow = wp.atomic_add(i_c, 1, 1)
    con_frictionless_id[irow] = i_con
    worldid_frictionless[irow] = w_id


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
    if m.opt.cone == types.MJ_CONE_ELLIPTIC:
      irow = wp.atomic_add(i_c, 2, condim)
      for j in range(condim):
        con_id[irow + j] = i_con
        con_dim_id[irow + j] = j
        worldid_con[irow + j] = w_id
    else:
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
    n_frictionless = d.ncon * d.nworld
    worldid_frictionless = wp.empty(n_frictionless, dtype=wp.int32)
    con_frictionless_id = wp.empty(n_frictionless, dtype=wp.int32)
    n_con = d.ncon * d.nworld * 12
    worldid_con = wp.empty(n_con, dtype=wp.int32)
    con_id = wp.empty(n_con, dtype=wp.int32)
    com_dim_id = wp.empty(n_con, dtype=wp.int32)
    wp.launch(
      _allocate_efc_contact_frictionless,
      dim=(d.nworld, d.ncon),
      inputs=[d, i_c],
      outputs=[worldid_frictionless, con_frictionless_id],
    )
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

    if (
      not (m.opt.disableflags & types.MJ_DSBL_EQUALITY)
      and m.efc_eq_connect_id.size != 0
    ):
      wp.launch(
        _efc_equality_connect,
        dim=(d.nworld, 3 * m.efc_eq_connect_id.size),
        inputs=[m, d, i_c, m.efc_eq_connect_id],
        outputs=[efcs],
      )

    if not (m.opt.disableflags & types.MJ_DSBL_EQUALITY) and m.efc_eq_weld_id.size != 0:
      wp.launch(
        _efc_equality_weld,
        dim=(d.nworld, 6 * m.efc_eq_weld_id.size),
        inputs=[m, d, i_c, m.efc_eq_weld_id],
        outputs=[efcs],
      )

    if (
      not (m.opt.disableflags & types.MJ_DSBL_EQUALITY) and m.efc_eq_joint_id.size != 0
    ):
      wp.launch(
        _efc_equality_joint,
        dim=(d.nworld, m.efc_eq_joint_id.size),
        inputs=[m, d, i_c, m.efc_eq_joint_id],
        outputs=[efcs],
      )

    if not (m.opt.disableflags & types.MJ_DSBL_FRICTIONLOSS) and (
      m.efc_dof_friction_id.size != 0
    ):
      wp.launch(
        _efc_dof_friction,
        dim=(d.nworld, m.efc_dof_friction_id.size),
        inputs=[m, i_c, m.efc_dof_friction_id],
        outputs=[efcs],
      )

    if not (m.opt.disableflags & types.MJ_DSBL_LIMIT) and (m.efc_jnt_ball_id.size != 0):
      wp.launch(
        _efc_limit_ball,
        dim=(d.nworld, m.efc_jnt_ball_id.size),
        inputs=[m, d, i_c, m.efc_jnt_ball_id],
        outputs=[efcs],
      )

    if not (m.opt.disableflags & types.MJ_DSBL_LIMIT) and (
      m.efc_jnt_slide_hinge_id.size != 0
    ):
      wp.launch(
        _efc_limit_slide_hinge,
        dim=(d.nworld, m.efc_jnt_slide_hinge_id.size),
        inputs=[m, d, i_c, m.efc_jnt_slide_hinge_id],
        outputs=[efcs],
      )

    con_frictionless_size = int(i_c.numpy()[1])
    if con_frictionless_size != 0:
      wp.launch(
        _efc_contact_frictionless,
        dim=con_frictionless_size,
        inputs=[m, d, worldid_frictionless, i_c, con_frictionless_id],
        outputs=[efcs],
      )

    con_size = int(i_c.numpy()[2])
    if con_size != 0:
      if m.opt.cone == types.MJ_CONE_ELLIPTIC:
        wp.launch(
          _efc_contact_elliptic,
          dim=con_size,
          inputs=[m, d, worldid_con, i_c, con_id, com_dim_id],
          outputs=[efcs],
        )
      else:
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
