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
from mujoco import mjx

MJ_MINVAL = mujoco.mjMINVAL
MJ_MINIMP = mujoco.mjMINIMP  # minimum constraint impedance
MJ_MAXIMP = mujoco.mjMAXIMP  # maximum constraint impedance
MJ_NREF = mujoco.mjNREF
MJ_NIMP = mujoco.mjNIMP


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


# joint types
MJ_JNT_FREE = mujoco.mjtJoint.mjJNT_FREE.value
MJ_JNT_BALL = mujoco.mjtJoint.mjJNT_BALL.value
MJ_JNT_SLIDE = mujoco.mjtJoint.mjJNT_SLIDE.value
MJ_JNT_HINGE = mujoco.mjtJoint.mjJNT_HINGE.value


# cone types
MJ_CONE_PYRAMIDAL = mujoco.mjtCone.mjCONE_PYRAMIDAL.value
MJ_CONE_ELLIPTIC = mujoco.mjtCone.mjCONE_ELLIPTIC.value


# equality cosntraint type
MJ_EQ_CONNECT = mujoco.mjtEq.mjEQ_CONNECT.value
MJ_EQ_WELD = mujoco.mjtEq.mjEQ_WELD.value
MJ_EQ_JOINT = mujoco.mjtEq.mjEQ_JOINT.value
MJ_EQ_TENDON = mujoco.mjtEq.mjEQ_TENDON.value


# object types
MJ_OBJ_UNKNOWN = mujoco.mjtObj.mjOBJ_UNKNOWN.value
MJ_OBJ_BODY = mujoco.mjtObj.mjOBJ_BODY.value
MJ_OBJ_XBODY = mujoco.mjtObj.mjOBJ_XBODY.value
MJ_OBJ_GEOM = mujoco.mjtObj.mjOBJ_GEOM.value
MJ_OBJ_SITE = mujoco.mjtObj.mjOBJ_SITE.value
MJ_OBJ_CAMERA = mujoco.mjtObj.mjOBJ_CAMERA.value


class vec10f(wp.types.vector(length=10, dtype=wp.float32)):
  pass


vec10 = vec10f
array2df = wp.array2d(dtype=wp.float32)
array3df = wp.array3d(dtype=wp.float32)


@wp.struct
class Option:
  gravity: wp.vec3
  is_sparse: bool  # warp only
  cone: int
  disableflags: int
  impratio: wp.float32
  timestep: float


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
  neq: int
  nmocap: int
  nM: int
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
  body_invweight0: wp.array(dtype=wp.float32, ndim=2)
  jnt_bodyid: wp.array(dtype=wp.int32, ndim=1)
  jnt_limited: wp.array(dtype=wp.int32, ndim=1)
  jnt_solref: wp.array(dtype=wp.float32, ndim=2)
  jnt_solimp: wp.array(dtype=wp.float32, ndim=2)
  jnt_type: wp.array(dtype=wp.int32, ndim=1)
  jnt_qposadr: wp.array(dtype=wp.int32, ndim=1)
  jnt_dofadr: wp.array(dtype=wp.int32, ndim=1)
  jnt_axis: wp.array(dtype=wp.vec3, ndim=1)
  jnt_pos: wp.array(dtype=wp.vec3, ndim=1)
  jnt_range: wp.array(dtype=wp.float32, ndim=2)
  jnt_margin: wp.array(dtype=wp.float32, ndim=1)
  jnt_stiffness: wp.array(dtype=wp.float32, ndim=1)
  geom_bodyid: wp.array(dtype=wp.int32, ndim=1)
  geom_pos: wp.array(dtype=wp.vec3, ndim=1)
  geom_quat: wp.array(dtype=wp.quat, ndim=1)
  site_bodyid: wp.array(dtype=wp.int32, ndim=1)
  site_pos: wp.array(dtype=wp.vec3, ndim=1)
  site_quat: wp.array(dtype=wp.quat, ndim=1)
  dof_bodyid: wp.array(dtype=wp.int32, ndim=1)
  dof_jntid: wp.array(dtype=wp.int32, ndim=1)
  dof_parentid: wp.array(dtype=wp.int32, ndim=1)
  dof_Madr: wp.array(dtype=wp.int32, ndim=1)
  dof_solref: wp.array(dtype=wp.float32, ndim=2)
  dof_solimp: wp.array(dtype=wp.float32, ndim=2)
  dof_frictionloss: wp.array(dtype=wp.float32, ndim=1)
  dof_armature: wp.array(dtype=wp.float32, ndim=1)
  dof_invweight0: wp.array(dtype=wp.float32, ndim=1)
  dof_damping: wp.array(dtype=wp.float32, ndim=1)
  actuator_actlimited: wp.array(dtype=wp.int32, ndim=1)
  actuator_actrange: wp.array(dtype=wp.float32, ndim=2)
  actuator_actadr: wp.array(dtype=wp.int32, ndim=1)
  actuator_dyntype: wp.array(dtype=wp.int32, ndim=1)
  actuator_dynprm: wp.array(dtype=wp.float32, ndim=2)
  eq_type: wp.array(dtype=wp.int32, ndim=1)
  eq_obj1id: wp.array(dtype=wp.int32, ndim=1)
  eq_obj2id: wp.array(dtype=wp.int32, ndim=1)
  eq_objtype: wp.array(dtype=wp.int32, ndim=1)
  eq_solref: wp.array(dtype=wp.float32, ndim=2)
  eq_solimp: wp.array(dtype=wp.float32, ndim=2)
  eq_data: wp.array(dtype=wp.float32, ndim=2)
  opt: Option
  # warp only
  efc_jnt_slide_hinge_id: wp.array(dtype=wp.int32, ndim=1)


@wp.struct
class Contact:
  dist: wp.array(dtype=wp.float32, ndim=2)
  pos: wp.array(dtype=wp.vec3f, ndim=2)
  frame: wp.array(dtype=wp.mat33f, ndim=2)
  includemargin: wp.array(dtype=wp.float32, ndim=2)
  friction: wp.array(dtype=wp.float32, ndim=3)
  solref: wp.array(dtype=wp.float32, ndim=3)
  solreffriction: wp.array(dtype=wp.float32, ndim=3)
  solimp: wp.array(dtype=wp.float32, ndim=3)
  dim: wp.array(dtype=wp.int32, ndim=2)
  geom: wp.array(dtype=wp.int32, ndim=3)
  efc_address: wp.array(dtype=wp.int32, ndim=2)


@wp.struct
class Data:
  nworld: int
  ncon: int
  ne: int
  nf: int
  nl: int
  nefc: int
  time: float
  qpos: wp.array(dtype=wp.float32, ndim=2)
  eq_active: wp.array(dtype=wp.int32, ndim=2)
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
  qfrc_bias: wp.array(dtype=wp.float32, ndim=2)
  qfrc_constraint: wp.array(dtype=wp.float32, ndim=2)
  qfrc_passive: wp.array(dtype=wp.float32, ndim=2)
  qfrc_spring: wp.array(dtype=wp.float32, ndim=2)
  qfrc_damper: wp.array(dtype=wp.float32, ndim=2)
  qfrc_actuator: wp.array(dtype=wp.float32, ndim=2)
  qfrc_smooth: wp.array(dtype=wp.float32, ndim=2)
  contact: Contact
  efc_J: wp.array(dtype=wp.float32, ndim=3)
  efc_pos: wp.array(dtype=wp.float32, ndim=2)
  efc_margin: wp.array(dtype=wp.float32, ndim=2)
  efc_frictionloss: wp.array(dtype=wp.float32, ndim=2)
  efc_D: wp.array(dtype=wp.float32, ndim=2)
  efc_aref: wp.array(dtype=wp.float32, ndim=2)

  # temp arrays
  qfrc_integration: wp.array(dtype=wp.float32, ndim=2)
  qacc_integration: wp.array(dtype=wp.float32, ndim=2)

  qM_integration: wp.array(dtype=wp.float32, ndim=3)
  qLD_integration: wp.array(dtype=wp.float32, ndim=3)
  qLDiagInv_integration: wp.array(dtype=wp.float32, ndim=2)
