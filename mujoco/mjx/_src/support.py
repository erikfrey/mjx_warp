import warp as wp
import mujoco
from . import types


def is_sparse(m: mujoco.MjModel):
  if m.opt.jacobian == mujoco.mjtJacobian.mjJAC_AUTO:
    return m.nv >= 60
  return m.opt.jacobian == mujoco.mjtJacobian.mjJAC_SPARSE


def mul_m(
  m: types.Model,
  d: types.Data,
  input: wp.array(ndim=2, dtype=wp.float32),
  output: wp.array(ndim=2, dtype=wp.float32),
):
  """Multiply vector by inertia matrix."""

  if not wp.static(m.opt.is_sparse):
    output.zero_()

    @wp.kernel
    def _matvec(
      mat: wp.array(ndim=3, dtype=wp.float32),
      input: wp.array(ndim=2, dtype=wp.float32),
      output: wp.array(ndim=2, dtype=wp.float32),
    ):
      worldid, rowid, colid = wp.tid()
      wp.atomic_add(
        output[worldid], rowid, mat[worldid, rowid, colid] * input[worldid, colid]
      )

    wp.launch(_matvec, dim=(d.nworld, m.nv, m.nv), inputs=[d.qM, input, output])
  else:
    # TODO(team): sparse version
    return
