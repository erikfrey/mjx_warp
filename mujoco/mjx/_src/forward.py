from typing import Optional

import warp as wp

from . import math
from . import smooth
from . import types


@wp.func
def quat_integrate_wp(q: wp.quat, v: wp.vec3, dt: wp.float32) -> wp.quat:
  """Integrates a quaternion given angular velocity and dt."""
  norm_ = wp.length(v)
  v = wp.normalize(v)  # does that need proper zero gradient handling?
  angle = dt * norm_

  q_res = math.axis_angle_to_quat(v, angle)
  q_res = math.mul_quat(q, q_res)

  return wp.normalize(q_res)


WarpMjMinVal = wp.constant(1e-15)


@wp.kernel
def next_activation(
    m: types.Model, d: types.Data, act_dot_in: wp.array2d(dtype=wp.float32)
):
  worldId, tid = wp.tid()

  # get the high/low range for each actuator state
  limited = m.actuator_actlimited[tid]
  range_low = wp.select(limited, -wp.inf, m.actuator_actrange[tid, 0])
  range_high = wp.select(limited, wp.inf, m.actuator_actrange[tid, 1])

  # get the actual actuation - skip if -1 (means stateless actuator)
  act_adr = m.actuator_actadr[tid]
  if act_adr == -1:
    return

  acts = d.act[worldId]
  acts_dot = act_dot_in[worldId]

  act = acts[act_adr]
  act_dot = acts_dot[act_adr]

  # check dynType
  dyn_type = m.actuator_dyntype[tid]
  dyn_prm = m.actuator_dynprm[tid, 0]

  # advance the actuation
  if dyn_type == 3:  # wp.static(WarpDynType.FILTEREXACT):
    tau = wp.select(
        dyn_prm < wp.static(WarpMjMinVal), dyn_prm, wp.static(WarpMjMinVal)
    )
    act = act + act_dot * tau * (1.0 - wp.exp(-m.timestep / tau))
  else:
    act = act + act_dot * m.timestep

  # apply limits
  wp.clamp(act, range_low, range_high)

  acts[act_adr] = act


@wp.kernel
def advance_velocities(
    m: types.Model, d: types.Data, qacc: wp.array2d(dtype=wp.float32)
):
  worldId, tid = wp.tid()
  d.qvel[worldId, tid] = d.qvel[worldId, tid] + qacc[worldId, tid] * m.timestep


@wp.kernel
def integrate_joint_positions(
    m: types.Model, d: types.Data, qvel_in: wp.array2d(dtype=wp.float32)
):
  worldId, tid = wp.tid()

  jnt_type = m.jnt_type[tid]
  qpos_adr = m.jnt_qposadr[tid]
  dof_adr = m.jnt_dofadr[tid]
  qpos = d.qpos[worldId]
  qvel = qvel_in[worldId]

  if jnt_type == 0:  # free joint
    qpos_pos = wp.vec3(qpos[qpos_adr], qpos[qpos_adr + 1], qpos[qpos_adr + 2])
    qvel_lin = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2])

    qpos_new = qpos_pos + m.timestep * qvel_lin

    qpos_quat = wp.quat(
        qpos[qpos_adr + 3],
        qpos[qpos_adr + 4],
        qpos[qpos_adr + 5],
        qpos[qpos_adr + 6],
    )
    qvel_ang = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5])

    qpos_quat_new = quat_integrate_wp(qpos_quat, qvel_ang, m.timestep)

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

    qpos_quat_new = quat_integrate_wp(qpos_quat, qvel_ang, m.timestep)

    qpos[qpos_adr] = qpos_quat_new[0]
    qpos[qpos_adr + 1] = qpos_quat_new[1]
    qpos[qpos_adr + 2] = qpos_quat_new[2]
    qpos[qpos_adr + 3] = qpos_quat_new[3]

  else:  # if jnt_type in (JointType.HINGE, JointType.SLIDE):
    qpos[qpos_adr] = qpos[qpos_adr] + m.timestep * qvel[dof_adr]


def _advance(
    m: types.Model,
    d: types.Data,
    act_dot: wp.array,
    qacc: wp.array,
    qvel: Optional[wp.array] = None,
) -> types.Data:
  """Advance state and time given activation derivatives and acceleration."""

  # skip if no stateful actuators.
  if m.na:
    wp.launch(next_activation, dim=(d.nworld, m.nu), inputs=[m, d, act_dot])

  wp.launch(advance_velocities, dim=(d.nworld, m.nv), inputs=[m, d, qacc])

  # advance positions with qvel if given, d.qvel otherwise (semi-implicit)
  if qvel is not None:
    qvel_in = qvel
  else:
    qvel_in = d.qvel

  wp.launch(
      integrate_joint_positions, dim=(d.nworld, m.njnt), inputs=[m, d, qvel_in]
  )

  d.time = d.time + m.timestep

  return d


def euler(m: types.Model, d: types.Data) -> types.Data:
  """Euler integrator, semi-implicit in velocity."""
  # integrate damping implicitly

  def add_damping_sum_qfrc(m: types.Model, d: types.Data, is_sparse: bool):

    @wp.kernel
    def add_damping_sum_qfrc_kernel(m: types.Model, d: types.Data):
      worldId, tid = wp.tid()

      if (wp.static(is_sparse)):
        dof_Madr = m.dof_Madr[tid]
        d.qM[worldId, 0, dof_Madr] += m.timestep * m.dof_damping[dof_Madr]
      else:
        d.qM[worldId, tid, tid] += m.timestep * m.dof_damping[tid]

      d.qfrc_eulerdamp[worldId, tid] = (
          d.qfrc_smooth[worldId, tid] + d.qfrc_constraint[worldId, tid]
      )
    
    wp.launch(add_damping_sum_qfrc_kernel, dim=(d.nworld, m.nv), inputs=[m, d])

  wp.copy(d.qacc_eulerdamp, d.qacc)

  if not m.opt.disableflags & types.MJ_DSBL_EULERDAMP:    
    add_damping_sum_qfrc(m, d, m.opt.is_sparse)
    smooth.factor_m(m, d)
    d.qacc_eulerdamp = smooth.solve_m(m, d, d.qfrc_eulerdamp, d.qacc_eulerdamp)
  return _advance(m, d, d.act_dot, d.qacc_eulerdamp)

def implicit(m: types.Model, d: types.Data) -> types.Data:
  """Integrates fully implicit in velocity."""
  
  @wp.kernel
  def actuator_bias_gain_vel(m: types.Model, d: types.Data):
    worldid, tid = wp.tid()

    bias_vel = 0.0
    gain_vel = 0.0

    actuator_biastype = m.actuator_biastype[tid]
    actuator_gaintype = m.actuator_gaintype[tid]
    actuator_dyntype = m.actuator_dyntype[tid]

    if actuator_biastype == BiasType.AFFINE:
      bias_vel = m.actuator_biasprm[tid, 2]

    if actuator_gaintype == GainType.AFFINE:
      gain_vel = m.actuator_gainprm[tid, 2]
    
    ctrl = d.ctrl[worldid, tid]

    if actuator_dyntype != DynType.NONE:
      ctrl = d.act[worldid, tid]

    vel[worldid, tid] = bias_vel + gain_vel * ctrl

  @wp.kernel
  def qderiv_add_damping(m: types.Model, d: types.Data, qderiv: wp.array3d(dtype=wp.float32)):
    worldid, tid = wp.tid()
    qderiv[worldid, tid, tid] = qderiv[worldid, tid, tid] - m.dof_damping[tid]

  @wp.kernel
  def subtract_qderiv_M(m: types.Model, m_temp: wp.array3d(dtype=wp.float32), qderiv: wp.array3d(dtype=wp.float32)):
    worldid, i, j = wp.tid()
    m_temp[worldid, i, j] = m_temp[worldid, i, j] - m.opt.timestep * qderiv[worldid, i, j]

  @wp.kernel
  def sum_qfrc_smooth_constraint(m: types.Model, d: types.Data, qfrc_out: wp.array(dtype=wp.float32)):
    worldid, tid = wp.tid()
    qfrc_out[worldid, tid] = d.qfrc_smooth[worldid, tid] + d.qfrc_constraint[worldid, tid]

  
  # do we need this here?
  qderiv = wp.zeros(shape=(m.nworld, m.nv, m.nv), dtype=wp.float32)
  qderiv_filled = False

  # qDeriv += d qfrc_actuator / d qvel
  if not m.opt.disableflags & types.MJ_DSBL_ACTUATION:
    vel = wp.zeros(shape=(m.nworld, m.nu), dtype=wp.float32) # todo: remove
    wp.launch(actuator_bias_gain_vel, dim=(m.nworld, m.nu), inputs=[m, d, vel])
    qderiv_filled = True

    qderiv = d.actuator_moment.T @ jp.diag(vel) @ d.actuator_moment

  # qDeriv += d qfrc_passive / d qvel
  if not m.opt.disableflags & types.MJ_DSBL_PASSIVE:
    # add damping to qderiv
    wp.launch(qderiv_add_damping, dim=(m.nworld, m.nv), inputs=[m, d, qderiv])
    qderiv_filled = True
    # TODO: tendon
    # TODO: fluid drag, not supported in MJX right now

  wp.clone(qacc, d.qacc) 

  if qderiv_filled:
    if (m.opt.is_sparse):
      pass #todo
    else:
      qm_temp = wp.clone(d.qM)
      wp.launch(subtract_qderiv_M, dim=(m.nworld, m.nv, m.nv), inputs=[m, qderiv, qm_temp])
    
    qfrc = wp.zeros(shape=(m.nworld, m.nv), dtype=wp.float32)
    qfrc = wp.launch(sum_qfrc_smooth_constraint, dim=(m.nworld, m.nv), inputs=[m, d, qfrc])
    qLD_temp = smooth.factor_m(m, d, qM_temp, qLD_temp)
    qacc = smooth.solve_m(m, d, dfrc, qacc)

  return _advance(m, d, d.act_dot, qacc)