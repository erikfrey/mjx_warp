import warp as wp
from . import passive
from . import smooth
from . import types


def fwd_position(m: types.Model, d: types.Data):
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


def fwd_velocity(m: types.Model, d: types.Data):
  """Velocity-dependent computations."""

  # TODO(team): tile operations?
  @wp.kernel
  def _actuator_velocity(d: types.Data):
    worldid, actid, dofid = wp.tid()
    moment = d.actuator_moment[worldid, actid]
    qvel = d.qvel[worldid]
    wp.atomic_add(d.actuator_velocity[worldid],
                  actid, moment[dofid] * qvel[dofid])

  wp.launch(_actuator_velocity, dim=(d.nworld, m.nu, m.nv), inputs=[d])

  smooth.com_vel(m, d)
  passive.passive(m, d)
  smooth.rne(m, d)


def fwd_acceleration(m: types.Model, d: types.Data):
  """Add up all non-constraint forces, compute qacc_smooth."""

  qfrc_applied = d.qfrc_applied
  # TODO(team) += support.xfrc_accumulate(m, d)

  @wp.kernel
  def _qfrc_smooth(d: types.Data, qfrc_applied: wp.array(ndim=2, dtype=wp.float32)):
    worldid, dofid = wp.tid()
    d.qfrc_smooth[worldid, dofid] = d.qfrc_passive[worldid, dofid] - d.qfrc_bias[worldid, dofid] + d.qfrc_actuator[worldid, dofid] + qfrc_applied[worldid, dofid]

  wp.launch(_qfrc_smooth, dim=(d.nworld, m.nv), inputs=[d, qfrc_applied])

  smooth.solve_m(m, d, d.qfrc_smooth, d.qacc_smooth)


def forward(m: types.Model, d: type.Data):
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
