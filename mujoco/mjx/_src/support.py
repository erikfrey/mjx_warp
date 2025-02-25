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
  res: wp.array(ndim=2, dtype=wp.float32),
  vec: wp.array(ndim=2, dtype=wp.float32),
):
  """Multiply vector by inertia matrix."""

  if not m.opt.is_sparse:
    res.zero_()

    @wp.kernel
    def _mul_m_dense(
      d: types.Data,
      res: wp.array(ndim=2, dtype=wp.float32),
      vec: wp.array(ndim=2, dtype=wp.float32),
    ):
      worldid, rowid, colid = wp.tid()
      wp.atomic_add(
        res[worldid], rowid, d.qM[worldid, rowid, colid] * vec[worldid, colid]
      )

    wp.launch(_mul_m_dense, dim=(d.nworld, m.nv, m.nv), inputs=[d, res, vec])
  else:

    @wp.kernel
    def _mul_m_sparse_diag(
      m: types.Model,
      d: types.Data,
      res: wp.array(ndim=2, dtype=wp.float32),
      vec: wp.array(ndim=2, dtype=wp.float32),
    ):
      worldid, dofid = wp.tid()
      res[worldid, dofid] = d.qM[worldid, 0, m.dof_Madr[dofid]] * vec[worldid, dofid]

    wp.launch(_mul_m_sparse_diag, dim=(d.nworld, m.nv), inputs=[m, d, res, vec])

    @wp.kernel
    def _mul_m_sparse_ij(
      m: types.Model,
      d: types.Data,
      res: wp.array(ndim=2, dtype=wp.float32),
      vec: wp.array(ndim=2, dtype=wp.float32),
    ):
      worldid, elementid = wp.tid()
      i = m.qM_i[elementid]
      j = m.qM_j[elementid]
      madr_ij = m.qM_madr_ij[elementid]

      qM = d.qM[worldid, 0, madr_ij]

      wp.atomic_add(res[worldid], i, qM * vec[worldid, j])
      wp.atomic_add(res[worldid], j, qM * vec[worldid, i])

    wp.launch(
      _mul_m_sparse_ij, dim=(d.nworld, m.qM_madr_ij.size), inputs=[m, d, res, vec]
    )
