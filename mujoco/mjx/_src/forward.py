import warp as wp
from . import types
from typing import Optional

@wp.func
def quat_mul_wp(u: wp.vec4, v: wp.vec4) -> wp.vec4:
  return wp.vec4(
      u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
      u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
      u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
      u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
  )

@wp.func
def quat_integrate_wp(q: wp.vec4, v: wp.vec3, dt: wp.float32) -> wp.vec4:
  """Integrates a quaternion given angular velocity and dt."""
  norm_ = wp.length(v)
  v = wp.normalize(v)  # does that need proper zero gradient handling?
  angle = dt * norm_

  # q_res = axis_angle_to_quat(v, angle)
  s, c = wp.sin(angle * 0.5), wp.cos(angle * 0.5)
  q_res = wp.vec4(c, s * v.x, s * v.y, s * v.z)

  # q_res = quat_mul(q, q_res)
  q_res = quat_mul_wp(q, q_res)

  return wp.normalize(q_res)

WarpMjMinVal = wp.constant(1e-15)


@wp.kernel
def next_activation(
    acts_: wp.array2d(dtype=wp.float32),
    acts_dot_: wp.array2d(dtype=wp.float32),
    acts_limited: wp.array(dtype=wp.uint8),
    acts_range: wp.array(dtype=wp.float32, ndim=2),
    acts_adr: wp.array(dtype=wp.int32),
    acts_dynType: wp.array(dtype=wp.int32),
    acts_dynPrm: wp.array(dtype=wp.float32, ndim=2),
    timestep: wp.float32,
):
  worldId, tid = wp.tid()

  # get the high/low range for each actuator state
  limited = acts_limited[tid]
  range_low = wp.select(limited, -wp.inf, acts_range[tid, 0])
  range_high = wp.select(limited, wp.inf, acts_range[tid, 1])

  # get the actual actuation - skip if -1 (means stateless actuator)
  act_adr = acts_adr[tid]
  if act_adr == -1:
    return
  
  acts = acts_[worldId]
  acts_dot = acts_dot_[worldId]

  act = acts[act_adr]
  act_dot = acts_dot[act_adr]

  # check dynType
  dyn_type = acts_dynType[tid]
  dyn_prm = acts_dynPrm[tid, 0]

  # advance the actuation
  if dyn_type == 3: #wp.static(WarpDynType.FILTEREXACT):
    tau = wp.select(dyn_prm < wp.static(WarpMjMinVal), dyn_prm, wp.static(WarpMjMinVal))
    act = act + act_dot * tau * (1.0 - wp.exp(-timestep / tau))
  else:
    act = act + act_dot * timestep

  # apply limits
  wp.clamp(act, range_low, range_high)

  acts[act_adr] = act


@wp.kernel
def advance_velocities(
    qvel: wp.array2d(dtype=wp.float32),
    qacc: wp.array2d(dtype=wp.float32),
    timestep: wp.float32,
):
  worldId, tid = wp.tid()
  qvel[worldId, tid] = qvel[worldId, tid] + qacc[worldId, tid] * timestep


@wp.kernel
def integrate_joint_positions(
    jnt_types: wp.array(dtype=wp.int32),
    jnt_qposadr: wp.array(dtype=wp.int32),
    jnt_dofadr: wp.array(dtype=wp.int32),
    timestep: wp.float32,
    qvel_: wp.array2d(dtype=wp.float32),
    qpos_: wp.array2d(dtype=wp.float32),
):
  worldId, tid = wp.tid()

  jnt_type = jnt_types[tid]
  qpos_adr = jnt_qposadr[tid]
  dof_adr = jnt_dofadr[tid]
  qpos = qpos_[worldId]
  qvel = qvel_[worldId]

  if jnt_type == 0: #wp.static(WarpJointType.FREE):
    qpos_pos = wp.vec3(qpos[qpos_adr], qpos[qpos_adr + 1], qpos[qpos_adr + 2])
    qvel_lin = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2])

    qpos_new = qpos_pos + timestep * qvel_lin

    qpos_quat = wp.vec4(
        qpos[qpos_adr + 3],
        qpos[qpos_adr + 4],
        qpos[qpos_adr + 5],
        qpos[qpos_adr + 6],
    )
    qvel_ang = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5])

    qpos_quat_new = quat_integrate_wp(qpos_quat, qvel_ang, timestep)

    qpos[qpos_adr] = qpos_new[0]
    qpos[qpos_adr + 1] = qpos_new[1]
    qpos[qpos_adr + 2] = qpos_new[2]
    qpos[qpos_adr + 3] = qpos_quat_new[0]
    qpos[qpos_adr + 4] = qpos_quat_new[1]
    qpos[qpos_adr + 5] = qpos_quat_new[2]
    qpos[qpos_adr + 6] = qpos_quat_new[3]

  elif jnt_type == 1: #wp.static(WarpJointType.BALL):
    qpos_quat = wp.vec4(
        qpos[qpos_adr],
        qpos[qpos_adr + 1],
        qpos[qpos_adr + 2],
        qpos[qpos_adr + 3],
    )
    qvel_ang = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2])

    qpos_quat_new = quat_integrate_wp(qpos_quat, qvel_ang, timestep)

    qpos[qpos_adr] = qpos_quat_new[0]
    qpos[qpos_adr + 1] = qpos_quat_new[1]
    qpos[qpos_adr + 2] = qpos_quat_new[2]
    qpos[qpos_adr + 3] = qpos_quat_new[3]

  else:  # if jnt_type in (JointType.HINGE, JointType.SLIDE):
    qpos[qpos_adr] = qpos[qpos_adr] + timestep * qvel[dof_adr]



def advance(
    m: types.Model, d: types.Data,
    act_dot: wp.array,
    qacc: wp.array,
    qvel: Optional[wp.array] = None,
) -> types.Data:
  """Advance state and time given activation derivatives and acceleration."""
  # skip if no stateful actuators.
  if m.na:
    # warp implementation of next activation - per actuator
    wp.launch(
        kernel=next_activation,
        dim=(d.nworld, m.nu),
        inputs=[
            d.act,  # (na)
            act_dot,  # (na)
            m.actuator_actlimited,  # (nu)
            m.actuator_actrange,  # (nu, 2)
            m.actuator_actadr,  # (nu)
            m.actuator_dyntype,  # (nu)
            m.actuator_dynprm,  # (nu, 10)
            m.timestep,  # todo switch to warp model
        ],
    )

  # warp implementation of velocity advancement - per dof
  wp.launch(
      kernel=advance_velocities, dim=(d.nworld, m.nv), inputs=[d.qvel, qacc, m.timestep]
  )

  # advance positions with qvel if given, d.qvel otherwise (semi-implicit)
  if qvel is not None:
    qvel_in = qvel
  else:
    qvel_in = d.qvel

  # warp implementation of integration - per joint
  wp.launch(
      kernel=integrate_joint_positions,
      dim=(d.nworld, m.njnt),
      inputs=[
          m.jnt_type,
          m.jnt_qposadr,
          m.jnt_dofadr,
          m.timestep,
          qvel_in,
          d.qpos,
      ],
  )

  d.time = d.time + m.timestep

  return d