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
import mujoco

from . import math
from . import passive
from . import smooth

from .types import array2df, array3df
from .types import Model
from .types import Data
from .types import MJ_MINVAL
from .types import DisableBit
from .types import JointType
from .types import DynType


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
    range_low = wp.select(limited, -wp.inf, m.actuator_actrange[actid][0])
    range_high = wp.select(limited, wp.inf, m.actuator_actrange[actid][1])

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
    dyn_prm = m.actuator_dynprm[actid][0]

    # advance the actuation
    if dyn_type == wp.static(DynType.FILTEREXACT.value):
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

    if jnt_type == wp.static(JointType.FREE.value):
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

    elif jnt_type == wp.static(JointType.BALL.value):  # ball joint
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

  if not m.opt.disableflags & DisableBit.EULERDAMP.value:
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
  smooth.transmission(m, d)


def fwd_velocity(m: Model, d: Data):
  """Velocity-dependent computations."""

  # TODO(team): tile operations?
  d.actuator_velocity.zero_()

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


def fwd_actuation(m: Model, d: Data):
  """Actuation-dependent computations."""
  if not m.nu:
    return

  # TODO support stateful actuators

  @wp.kernel
  def _force(
    m: Model,
    ctrl: array2df,
    # outputs
    force: array2df,
  ):
    worldid, dofid = wp.tid()
    gain = m.actuator_gainprm[dofid, 0]
    bias = m.actuator_biasprm[dofid, 0]
    # TODO support gain types other than FIXED
    c = ctrl[worldid, dofid]
    if m.actuator_ctrllimited[dofid]:
      r = m.actuator_ctrlrange[dofid]
      c = wp.clamp(c, r[0], r[1])
    f = gain * c + bias
    if m.actuator_forcelimited[dofid]:
      r = m.actuator_forcerange[dofid]
      f = wp.clamp(f, r[0], r[1])
    force[worldid, dofid] = f

  wp.launch(
    _force, dim=[d.nworld, m.nu], inputs=[m, d.ctrl], outputs=[d.actuator_force]
  )

  @wp.kernel
  def _qfrc(m: Model, moment: array3df, force: array2df, qfrc: array2df):
    worldid, vid = wp.tid()

    s = float(0.0)
    for uid in range(m.nu):
      # TODO consider using Tile API or transpose moment for better access pattern
      s += moment[worldid, uid, vid] * force[worldid, uid]
    jntid = m.dof_jntid[vid]
    if m.jnt_actfrclimited[jntid]:
      r = m.jnt_actfrcrange[jntid]
      s = wp.clamp(s, r[0], r[1])
    qfrc[worldid, vid] = s

  wp.launch(
    _qfrc,
    dim=(d.nworld, m.nv),
    inputs=[m, d.actuator_moment, d.actuator_force],
    outputs=[d.qfrc_actuator],
  )

  # TODO actuator-level gravity compensation, skip if added as passive force

  return d


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
  fwd_velocity(m, d)
  # TODO(team): sensor.sensor_vel
  fwd_actuation(m, d)
  fwd_acceleration(m, d)
  # TODO(team): sensor.sensor_acc

  # if nefc == 0
  wp.copy(d.qacc, d.qacc_smooth)

  # TODO(team): solver.solve


def step(m: Model, d: Data):
  """Advance simulation."""
  forward(m, d)

  if m.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER:
    euler(m, d)
  elif m.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
    # TODO(team): rungekutta4
    raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")
  elif m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
    # TODO(team): implicit
    raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")
  else:
    raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")
