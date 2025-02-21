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

MJ_MINVAL = mujoco.mjMINVAL

# disable flags - TODO(team): make this bullet-proof.
MJ_DSBL_CONSTRAINT = mujoco.mjtDisableBit.mjDSBL_CONSTRAINT.value
MJ_DSBL_EQUALITY = mujoco.mjtDisableBit.mjDSBL_EQUALITY.value
MJ_DSBL_FRICTIONLOSS = mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS.value
MJ_DSBL_LIMIT = mujoco.mjtDisableBit.mjDSBL_LIMIT.value
MJ_DSBL_CONTACT = mujoco.mjtDisableBit.mjDSBL_CONTACT.value
MJ_DSBL_PASSIVE = mujoco.mjtDisableBit.mjDSBL_PASSIVE.value
MJ_DSBL_GRAVITY = mujoco.mjtDisableBit.mjDSBL_GRAVITY.value
MJ_DSBL_CLAMPCTRL = mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL.value
MJ_DSBL_WARMSTART = mujoco.mjtDisableBit.mjDSBL_WARMSTART.value
MJ_DSNL_FILTERPARENT = mujoco.mjtDisableBit.mjDSBL_FILTERPARENT.value
MJ_DSBL_ACTUATION = mujoco.mjtDisableBit.mjDSBL_ACTUATION.value
MJ_DSBL_REFSAFE = mujoco.mjtDisableBit.mjDSBL_REFSAFE.value
MJ_DSBL_SENSOR = mujoco.mjtDisableBit.mjDSBL_SENSOR.value
MJ_DSBL_MIDPHASE = mujoco.mjtDisableBit.mjDSBL_MIDPHASE.value
MJ_DSBL_EULERDAMP = mujoco.mjtDisableBit.mjDSBL_EULERDAMP.value


class vec5f(wp.types.vector(length=5, dtype=wp.float32)):
  pass

class vec10f(wp.types.vector(length=10, dtype=wp.float32)):
  pass


vec10 = vec10f
array2df = wp.array2d(dtype=wp.float32, ndim=2)

@wp.struct
class Option:
  gravity: wp.vec3
  is_sparse: bool  # warp only
  timestep: float
  disableflags: int


@wp.struct
class Model:
  nq: int
  nv: int
  na: int
  nu: int
  nbody: int
  njnt: int
  ngeom: int
  nsite: int
  nmocap: int
  nM: int
  nconmax: int
  opt: Option
  qpos0: wp.array(dtype=wp.float32, ndim=1)
  qpos_spring: wp.array(dtype=wp.float32, ndim=1)
  body_tree: wp.array(dtype=wp.int32, ndim=1)  # warp only
  body_treeadr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_update_tree: wp.array(dtype=wp.vec3i, ndim=1)  # warp only
  qLD_update_treeadr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_tile: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_tileadr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_tilesize: wp.array(dtype=wp.int32, ndim=1)  # warp only
  body_dofadr: wp.array(dtype=wp.int32, ndim=1)
  body_dofnum: wp.array(dtype=wp.int32, ndim=1)
  body_jntadr: wp.array(dtype=wp.int32, ndim=1)
  body_jntnum: wp.array(dtype=wp.int32, ndim=1)
  body_parentid: wp.array(dtype=wp.int32, ndim=1)
  body_mocapid: wp.array(dtype=wp.int32, ndim=1)
  body_pos: wp.array(dtype=wp.vec3, ndim=1)
  body_quat: wp.array(dtype=wp.quat, ndim=1)
  body_ipos: wp.array(dtype=wp.vec3, ndim=1)
  body_iquat: wp.array(dtype=wp.quat, ndim=1)
  body_rootid: wp.array(dtype=wp.int32, ndim=1)
  body_inertia: wp.array(dtype=wp.vec3, ndim=1)
  body_mass: wp.array(dtype=wp.float32, ndim=1)
  jnt_bodyid: wp.array(dtype=wp.int32, ndim=1)
  jnt_type: wp.array(dtype=wp.int32, ndim=1)
  jnt_qposadr: wp.array(dtype=wp.int32, ndim=1)
  jnt_dofadr: wp.array(dtype=wp.int32, ndim=1)
  jnt_axis: wp.array(dtype=wp.vec3, ndim=1)
  jnt_pos: wp.array(dtype=wp.vec3, ndim=1)
  jnt_stiffness: wp.array(dtype=wp.float32, ndim=1)
  geom_pos: wp.array(dtype=wp.vec3, ndim=1)
  geom_quat: wp.array(dtype=wp.quat, ndim=1)
  site_pos: wp.array(dtype=wp.vec3, ndim=1)
  site_quat: wp.array(dtype=wp.quat, ndim=1)
  dof_bodyid: wp.array(dtype=wp.int32, ndim=1)
  dof_jntid: wp.array(dtype=wp.int32, ndim=1)
  dof_parentid: wp.array(dtype=wp.int32, ndim=1)
  dof_Madr: wp.array(dtype=wp.int32, ndim=1)
  dof_armature: wp.array(dtype=wp.float32, ndim=1)
  dof_damping: wp.array(dtype=wp.float32, ndim=1)
  actuator_actlimited: wp.array(dtype=wp.int32, ndim=1)
  actuator_actrange: wp.array(dtype=wp.float32, ndim=2)
  actuator_actadr: wp.array(dtype=wp.int32, ndim=1)
  actuator_dyntype: wp.array(dtype=wp.int32, ndim=1)
  actuator_dynprm: wp.array(dtype=wp.float32, ndim=2)


@wp.struct
class Contact:
   worldid: wp.array1d(dtype=int)
   geom: wp.array1d(wp.vec2i)
   efc_address: wp.array1d(dtype=int)
   dist: wp.array1d(dtype=float)
   pos: wp.array1d(dtype=wp.vec3)
   frame: wp.array1d(dtype=wp.mat33)
   includemargin: wp.array1d(dtype=float)
   friction: wp.array1d(dtype=vec5f)
   solref: wp.array1d(dtype=wp.vec2)
   solreffriction: wp.array1d(dtype=wp.vec2)
   solimp: wp.array1d(dtype=vec5f)
   dim: wp.array1d(dtype=int)


@wp.struct
class Data:
  nworld: int
  nconmax: int
  ncon: wp.array(dtype=wp.int32, ndim=1)
  ncon_total: wp.array(dtype=wp.int32, ndim=1)  # warp only
  time: wp.array(dtype=wp.float32, ndim=1)
  qpos: wp.array(dtype=wp.float32, ndim=2)
  qvel: wp.array(dtype=wp.float32, ndim=2)
  qfrc_applied: wp.array(dtype=wp.float32, ndim=2)
  mocap_pos: wp.array(dtype=wp.vec3, ndim=2)
  mocap_quat: wp.array(dtype=wp.quat, ndim=2)
  qacc: wp.array(dtype=wp.float32, ndim=2)
  qacc_smooth: wp.array(dtype=wp.float32, ndim=2)
  xanchor: wp.array(dtype=wp.vec3, ndim=2)
  xaxis: wp.array(dtype=wp.vec3, ndim=2)
  xmat: wp.array(dtype=wp.mat33, ndim=2)
  xpos: wp.array(dtype=wp.vec3, ndim=2)
  xquat: wp.array(dtype=wp.quat, ndim=2)
  xipos: wp.array(dtype=wp.vec3, ndim=2)
  ximat: wp.array(dtype=wp.mat33, ndim=2)
  subtree_com: wp.array(dtype=wp.vec3, ndim=2)
  geom_xpos: wp.array(dtype=wp.vec3, ndim=2)
  geom_xmat: wp.array(dtype=wp.mat33, ndim=2)
  site_xpos: wp.array(dtype=wp.vec3, ndim=2)
  site_xmat: wp.array(dtype=wp.mat33, ndim=2)
  cinert: wp.array(dtype=vec10, ndim=2)
  cdof: wp.array(dtype=wp.spatial_vector, ndim=2)
  actuator_moment: wp.array(dtype=wp.float32, ndim=3)
  crb: wp.array(dtype=vec10, ndim=2)
  qM: wp.array(dtype=wp.float32, ndim=3)
  qLD: wp.array(dtype=wp.float32, ndim=3)
  act: wp.array(dtype=wp.float32, ndim=2)
  act_dot: wp.array(dtype=wp.float32, ndim=2)
  qLDiagInv: wp.array(dtype=wp.float32, ndim=2)
  actuator_velocity: wp.array(dtype=wp.float32, ndim=2)
  cvel: wp.array(dtype=wp.spatial_vector, ndim=2)
  cdof_dot: wp.array(dtype=wp.spatial_vector, ndim=2)
  contact: Contact
  qfrc_bias: wp.array(dtype=wp.float32, ndim=2)
  qfrc_constraint: wp.array(dtype=wp.float32, ndim=2)
  qfrc_passive: wp.array(dtype=wp.float32, ndim=2)
  qfrc_spring: wp.array(dtype=wp.float32, ndim=2)
  qfrc_damper: wp.array(dtype=wp.float32, ndim=2)
  qfrc_actuator: wp.array(dtype=wp.float32, ndim=2)
  qfrc_smooth: wp.array(dtype=wp.float32, ndim=2)

  # temp arrays
  qfrc_integration: wp.array(dtype=wp.float32, ndim=2)
  qacc_integration: wp.array(dtype=wp.float32, ndim=2)

  qM_integration: wp.array(dtype=wp.float32, ndim=3)
  qLD_integration: wp.array(dtype=wp.float32, ndim=3)
  qLDiagInv_integration: wp.array(dtype=wp.float32, ndim=2)
