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

from typing import Optional

import warp as wp

from . import math
from . import passive
from . import smooth

from .types import array2df
from .types import Model
from .types import Data
from .types import MJ_MINVAL
from .types import MJ_DSBL_EULERDAMP
from .types import MJ_DSBL_ACTUATION
from .types import MJ_DSBL_PASSIVE
from .types import MJ_BIASTYPE_AFFINE
from .types import MJ_GAINTYPE_AFFINE
from .types import MJ_DYNTYPE_NONE


def _advance(
  m: Model,
  d: Data,
  act_dot: wp.array,
  qacc: wp.array,
  qvel: Optional[wp.array] = None,
) -> Data:
  """Advance state and time given activation derivatives and acceleration."""

  # TODO(team): can we assume static timesteps?

  @wp.kernel
  def next_activation(
    m: Model,
    d: Data,
    act_dot_in: array2df,
  ):
    worldId, actid = wp.tid()

    # get the high/low range for each actuator state
    limited = m.actuator_actlimited[actid]
    range_low = wp.select(limited, -wp.inf, m.actuator_actrange[actid, 0])
    range_high = wp.select(limited, wp.inf, m.actuator_actrange[actid, 1])

    # get the actual actuation - skip if -1 (means stateless actuator)
    act_adr = m.actuator_actadr[actid]
    if act_adr == -1:
      return

    acts = d.act[worldId]
    acts_dot = act_dot_in[worldId]

    act = acts[act_adr]
    act_dot = acts_dot[act_adr]

    # check dynType
    dyn_type = m.actuator_dyntype[actid]
    dyn_prm = m.actuator_dynprm[actid, 0]

    # advance the actuation
    if dyn_type == 3:  # wp.static(WarpDynType.FILTEREXACT):
      tau = wp.select(dyn_prm < MJ_MINVAL, dyn_prm, MJ_MINVAL)
      act = act + act_dot * tau * (1.0 - wp.exp(-m.opt.timestep / tau))
    else:
      act = act + act_dot * m.opt.timestep

    # apply limits
    wp.clamp(act, range_low, range_high)

    acts[act_adr] = act

  @wp.kernel
  def advance_velocities(m: Model, d: Data, qacc: array2df):
    worldId, tid = wp.tid()
    d.qvel[worldId, tid] = d.qvel[worldId, tid] + qacc[worldId, tid] * m.opt.timestep

  @wp.kernel
  def integrate_joint_positions(m: Model, d: Data, qvel_in: array2df):
    worldId, jntid = wp.tid()

    jnt_type = m.jnt_type[jntid]
    qpos_adr = m.jnt_qposadr[jntid]
    dof_adr = m.jnt_dofadr[jntid]
    qpos = d.qpos[worldId]
    qvel = qvel_in[worldId]

    if jnt_type == 0:  # free joint
      qpos_pos = wp.vec3(qpos[qpos_adr], qpos[qpos_adr + 1], qpos[qpos_adr + 2])
      qvel_lin = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2])

      qpos_new = qpos_pos + m.opt.timestep * qvel_lin

      qpos_quat = wp.quat(
        qpos[qpos_adr + 3],
        qpos[qpos_adr + 4],
        qpos[qpos_adr + 5],
        qpos[qpos_adr + 6],
      )
      qvel_ang = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5])

      qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, m.opt.timestep)

      qpos[qpos_adr] = qpos_new[0]
      qpos[qpos_adr + 1] = qpos_new[1]
      qpos[qpos_adr + 2] = qpos_new[2]
      qpos[qpos_adr + 3] = qpos_quat_new[0]
      qpos[qpos_adr + 4] = qpos_quat_new[1]
      qpos[qpos_adr + 5] = qpos_quat_new[2]
      qpos[qpos_adr + 6] = qpos_quat_new[3]

    elif jnt_type == 1:  # ball joint
      qpos_quat = wp.quat(
        qpos[qpos_adr],
        qpos[qpos_adr + 1],
        qpos[qpos_adr + 2],
        qpos[qpos_adr + 3],
      )
      qvel_ang = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2])

      qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, m.opt.timestep)

      qpos[qpos_adr] = qpos_quat_new[0]
      qpos[qpos_adr + 1] = qpos_quat_new[1]
      qpos[qpos_adr + 2] = qpos_quat_new[2]
      qpos[qpos_adr + 3] = qpos_quat_new[3]

    else:  # if jnt_type in (JointType.HINGE, JointType.SLIDE):
      qpos[qpos_adr] = qpos[qpos_adr] + m.opt.timestep * qvel[dof_adr]

  # skip if no stateful actuators.
  if m.na:
    wp.launch(next_activation, dim=(d.nworld, m.nu), inputs=[m, d, act_dot])

  wp.launch(advance_velocities, dim=(d.nworld, m.nv), inputs=[m, d, qacc])

  # advance positions with qvel if given, d.qvel otherwise (semi-implicit)
  if qvel is not None:
    qvel_in = qvel
  else:
    qvel_in = d.qvel

  wp.launch(integrate_joint_positions, dim=(d.nworld, m.njnt), inputs=[m, d, qvel_in])

  d.time = d.time + m.opt.timestep
  return d


def euler(m: Model, d: Data) -> Data:
  """Euler integrator, semi-implicit in velocity."""
  # integrate damping implicitly

  def add_damping_sum_qfrc(m: Model, d: Data, is_sparse: bool):
    @wp.kernel
    def add_damping_sum_qfrc_kernel_sparse(m: Model, d: Data):
      worldId, tid = wp.tid()

      dof_Madr = m.dof_Madr[tid]
      d.qM_integration[worldId, 0, dof_Madr] += m.opt.timestep * m.dof_damping[dof_Madr]

      d.qfrc_integration[worldId, tid] = (
        d.qfrc_smooth[worldId, tid] + d.qfrc_constraint[worldId, tid]
      )

    @wp.kernel
    def add_damping_sum_qfrc_kernel_dense(m: Model, d: Data):
      worldid, i, j = wp.tid()

      damping = wp.select(i == j, 0.0, m.opt.timestep * m.dof_damping[i])
      d.qM_integration[worldid, i, j] = d.qM[worldid, i, j] + damping

      if i == 0:
        d.qfrc_integration[worldid, j] = (
          d.qfrc_smooth[worldid, j] + d.qfrc_constraint[worldid, j]
        )

    if is_sparse:
      wp.copy(d.qM_integration, d.qM)
      wp.launch(add_damping_sum_qfrc_kernel_sparse, dim=(d.nworld, m.nv), inputs=[m, d])
    else:
      wp.launch(
        add_damping_sum_qfrc_kernel_dense, dim=(d.nworld, m.nv, m.nv), inputs=[m, d]
      )

  if not m.opt.disableflags & MJ_DSBL_EULERDAMP:
    add_damping_sum_qfrc(m, d, m.opt.is_sparse)
    smooth.factor_i(m, d, d.qM_integration, d.qLD_integration, d.qLDiagInv_integration)
    smooth.solve_LD(
      m,
      d,
      d.qLD_integration,
      d.qLDiagInv_integration,
      d.qacc_integration,
      d.qfrc_integration,
    )
    return _advance(m, d, d.act_dot, d.qacc_integration)

  return _advance(m, d, d.act_dot, d.qacc)


def implicit(m: Model, d: Data) -> Data:
  """Integrates fully implicit in velocity."""

  # we might be able to exploit some sparsity here -
  # although setting to 0 is still needed somewhere.
  #
  # we only need to load ctrl if gain_vel is non zero.
  # might make more sense to dispatch the load though
  # and throw it away later, because otherwise you'll
  # just introduce a tight dependency.
  #
  @wp.kernel
  def actuator_bias_gain_vel(m: Model, d: Data, vel: array2df):
    worldid, tid = wp.tid()

    bias_vel = 0.0
    gain_vel = 0.0

    actuator_biastype = m.actuator_biastype[tid]
    actuator_gaintype = m.actuator_gaintype[tid]
    actuator_dyntype = m.actuator_dyntype[tid]

    if actuator_biastype == MJ_BIASTYPE_AFFINE:
      bias_vel = m.actuator_biasprm[tid, 2]

    if actuator_gaintype == MJ_GAINTYPE_AFFINE:
      gain_vel = m.actuator_gainprm[tid, 2]

    ctrl = d.ctrl[worldid, tid]

    if actuator_dyntype != MJ_DYNTYPE_NONE:
      ctrl = d.act[worldid, tid]

    vel[worldid, tid] = bias_vel + gain_vel * ctrl

  @wp.kernel
  def qderiv_add_damping(
    m: Model, d: Data, qderiv: wp.array3d(dtype=wp.float32)
  ):
    worldid, tid = wp.tid()
    qderiv[worldid, tid, tid] = qderiv[worldid, tid, tid] - m.dof_damping[tid]

  @wp.kernel
  def subtract_qderiv_M(
    m: Model,
    qderiv: wp.array3d(dtype=wp.float32),
    qM_integration: wp.array3d(dtype=wp.float32),
  ):
    worldid, i, j = wp.tid()
    qM_integration[worldid, i, j] = (
      qM_integration[worldid, i, j] - m.opt.timestep * qderiv[worldid, i, j]
    )

  @wp.kernel
  def sum_qfrc_smooth_constraint(m: Model, d: Data):
    worldid, tid = wp.tid()
    d.qfrc_integration[worldid, tid] = (
      d.qfrc_smooth[worldid, tid] + d.qfrc_constraint[worldid, tid]
    )

  # do we need this here?
  qderiv = wp.zeros(shape=(d.nworld, m.nv, m.nv), dtype=wp.float32)
  qderiv_filled = False

  # qDeriv += d qfrc_actuator / d qvel
  if not wp.static(m.opt.disableflags & MJ_DSBL_ACTUATION):
    vel = wp.zeros(shape=(d.nworld, m.nu), dtype=wp.float32)  # todo: remove
    wp.launch(actuator_bias_gain_vel, dim=(d.nworld, m.nu), inputs=[m, d, vel])
    qderiv_filled = True

    qderiv = d.actuator_moment.T @ jp.diag(vel) @ d.actuator_moment

  # qDeriv += d qfrc_passive / d qvel
  if not m.opt.disableflags & types.MJ_DSBL_PASSIVE:
    # add damping to qderiv
    wp.launch(qderiv_add_damping, dim=(m.nworld, m.nv), inputs=[m, d, qderiv])
    qderiv_filled = True
    # TODO: tendon
    # TODO: fluid drag, not supported in MJX right now

  wp.copy(d.qacc_integration, d.qacc)

  if qderiv_filled:
    if m.opt.is_sparse:
      pass  # todo
    else:
      wp.copy(d.qM_integration, d.qM)
      wp.launch(
        subtract_qderiv_M,
        dim=(m.nworld, m.nv, m.nv),
        inputs=[m, qderiv, d.qM_integration],
      )

    wp.launch(sum_qfrc_smooth_constraint, dim=(m.nworld, m.nv), inputs=[m, d])
    smooth.factor_m(m, d, d.qM_integration, d.qLD_integration, d.qLDiagInv_integration)
    smooth.solve_m(
      m,
      d,
      d.qLD_integration,
      d.qLDiagInv_integration,
      d.qfrc_integration,
      d.qacc_integration,
    )

  return _advance(m, d, d.act_dot, d.qacc_integration)


def fwd_position(m: Model, d: Data):
  """Position-dependent computations."""

  smooth.kinematics(m, d)
  smooth.com_pos(m, d)
  # TODO(team): smooth.camlight
  # TODO(team): smooth.tendon
  smooth.crb(m, d)
  smooth.factor_m(m, d)
  # TODO(team): collision_driver.collision
  # TODO(team): constraint.make_constraint
  # TODO(team): smooth.transmission


def fwd_velocity(m: Model, d: Data):
  """Velocity-dependent computations."""

  # TODO(team): tile operations?
  @wp.kernel
  def _actuator_velocity(d: Data):
    worldid, actid, dofid = wp.tid()
    moment = d.actuator_moment[worldid, actid]
    qvel = d.qvel[worldid]
    wp.atomic_add(d.actuator_velocity[worldid], actid, moment[dofid] * qvel[dofid])

  wp.launch(_actuator_velocity, dim=(d.nworld, m.nu, m.nv), inputs=[d])

  smooth.com_vel(m, d)
  passive.passive(m, d)
  smooth.rne(m, d)


def fwd_acceleration(m: Model, d: Data):
  """Add up all non-constraint forces, compute qacc_smooth."""

  qfrc_applied = d.qfrc_applied
  # TODO(team) += support.xfrc_accumulate(m, d)

  @wp.kernel
  def _qfrc_smooth(d: Data, qfrc_applied: wp.array(ndim=2, dtype=wp.float32)):
    worldid, dofid = wp.tid()
    d.qfrc_smooth[worldid, dofid] = (
      d.qfrc_passive[worldid, dofid]
      - d.qfrc_bias[worldid, dofid]
      + d.qfrc_actuator[worldid, dofid]
      + qfrc_applied[worldid, dofid]
    )

  wp.launch(_qfrc_smooth, dim=(d.nworld, m.nv), inputs=[d, qfrc_applied])

  smooth.solve_m(m, d, d.qacc_smooth, d.qfrc_smooth)


def forward(m: Model, d: Data):
  """Forward dynamics."""

  fwd_position(m, d)
  # TODO(team): sensor.sensor_pos
  # TODO(taylorhowell): fwd_velocity
  # TODO(team): sensor.sensor_vel
  # TODO(team): fwd_actuation
  fwd_acceleration(m, d)
  # TODO(team): sensor.sensor_acc

  # if nefc == 0
  wp.copy(d.qacc, d.qacc_smooth)

  # TODO(team): solver.solve
