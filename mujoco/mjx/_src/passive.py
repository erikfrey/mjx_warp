import warp as wp

from . import math
from .types import Model
from .types import Data


def passive(m: Model, d: Data):
  """Adds all passive forces."""

  @wp.kernel
  def _spring(m: Model, d: Data):
    worldid, jntid = wp.tid()
    stiffness = m.jnt_stiffness[jntid]
    dofid = m.jnt_dofadr[jntid]

    if stiffness == 0.0:
      return

    jnt_type = m.jnt_type[jntid]
    qposid = m.jnt_qposadr[jntid]

    if jnt_type == 0:  # mjJNT_FREE
      dif = wp.vec3(
        d.qpos[worldid, qposid + 0] - m.qpos_spring[qposid + 0],
        d.qpos[worldid, qposid + 1] - m.qpos_spring[qposid + 1],
        d.qpos[worldid, qposid + 2] - m.qpos_spring[qposid + 2],
      )
      d.qfrc_spring[worldid, dofid + 0] = -stiffness * dif[0]
      d.qfrc_spring[worldid, dofid + 1] = -stiffness * dif[1]
      d.qfrc_spring[worldid, dofid + 2] = -stiffness * dif[2]
      rot = wp.quat(
        d.qpos[worldid, qposid + 3],
        d.qpos[worldid, qposid + 4],
        d.qpos[worldid, qposid + 5],
        d.qpos[worldid, qposid + 6],
      )
      ref = wp.quat(
        m.qpos_spring[qposid + 3],
        m.qpos_spring[qposid + 4],
        m.qpos_spring[qposid + 5],
        m.qpos_spring[qposid + 6],
      )
      dif = math.quat_sub(rot, ref)
      d.qfrc_spring[worldid, dofid + 3] = -stiffness * dif[0]
      d.qfrc_spring[worldid, dofid + 4] = -stiffness * dif[1]
      d.qfrc_spring[worldid, dofid + 5] = -stiffness * dif[2]
    elif jnt_type == 1:  # mjJNT_BALL
      rot = wp.quat(
        d.qpos[worldid, qposid + 0],
        d.qpos[worldid, qposid + 1],
        d.qpos[worldid, qposid + 2],
        d.qpos[worldid, qposid + 3],
      )
      ref = wp.quat(
        m.qpos_spring[qposid + 0],
        m.qpos_spring[qposid + 1],
        m.qpos_spring[qposid + 2],
        m.qpos_spring[qposid + 3],
      )
      dif = math.quat_sub(rot, ref)
      d.qfrc_spring[worldid, dofid + 0] = -stiffness * dif[0]
      d.qfrc_spring[worldid, dofid + 1] = -stiffness * dif[1]
      d.qfrc_spring[worldid, dofid + 2] = -stiffness * dif[2]
    elif jnt_type == 2 or jnt_type == 3:  # mjJNT_SLIDE, mjJNT_HINGE
      fdif = d.qpos[worldid, qposid] - m.qpos_spring[qposid]
      d.qfrc_spring[worldid, dofid] = -stiffness * fdif

  @wp.kernel
  def _damper_passive(m: Model, d: Data):
    worldid, dofid = wp.tid()
    damping = m.dof_damping[dofid]
    qfrc_damper = -damping * d.qvel[worldid, dofid]

    d.qfrc_damper[worldid, dofid] = qfrc_damper
    d.qfrc_passive[worldid, dofid] = qfrc_damper + d.qfrc_spring[worldid, dofid]

  # TODO(team): mj_gravcomp
  # TODO(team): mj_ellipsoidFluidModel
  # TODO(team): mj_inertiaBoxFluidModell

  d.qfrc_spring.zero_()
  wp.launch(_spring, dim=(d.nworld, m.njnt), inputs=[m, d])
  wp.launch(_damper_passive, dim=(d.nworld, m.nv), inputs=[m, d])
