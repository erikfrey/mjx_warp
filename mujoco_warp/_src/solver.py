# Copyright 2025 The Newton Developers
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

import mujoco
import warp as wp

from . import smooth
from . import support
from . import types
from .warp_util import event_scope
from .warp_util import kernel
from .warp_util import kernel_copy


def _create_context(m: types.Model, d: types.Data, grad: bool = True):
  @kernel
  def _init_context(d: types.Data):
    worldid = wp.tid()
    d.efc.cost[worldid] = wp.inf
    d.efc.solver_niter[worldid] = 0
    d.efc.done[worldid] = 0
    if grad:
      d.efc.search_dot[worldid] = 0.0

  @kernel
  def _jaref(m: types.Model, d: types.Data):
    efcid, dofid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc.worldid[efcid]
    wp.atomic_add(
      d.efc.Jaref,
      efcid,
      d.efc.J[efcid, dofid] * d.qacc[worldid, dofid] - d.efc.aref[efcid] / float(m.nv),
    )

  @kernel
  def _search(d: types.Data):
    worldid, dofid = wp.tid()
    search = -1.0 * d.efc.Mgrad[worldid, dofid]
    d.efc.search[worldid, dofid] = search
    wp.atomic_add(d.efc.search_dot, worldid, search * search)

  wp.launch(_init_context, dim=(d.nworld), inputs=[d])

  # jaref = d.efc_J @ d.qacc - d.efc_aref
  d.efc.Jaref.zero_()

  wp.launch(_jaref, dim=(d.njmax, m.nv), inputs=[m, d])

  # Ma = qM @ qacc
  support.mul_m(m, d, d.efc.Ma, d.qacc)

  _update_constraint(m, d)
  if grad:
    _update_gradient(m, d)

    # search = -Mgrad
    wp.launch(_search, dim=(d.nworld, m.nv), inputs=[d])


def _update_constraint(m: types.Model, d: types.Data):
  @kernel
  def _init_cost(d: types.Data):
    worldid = wp.tid()
    d.efc.prev_cost[worldid] = d.efc.cost[worldid]
    d.efc.cost[worldid] = 0.0
    d.efc.gauss[worldid] = 0.0

  @kernel
  def _efc_kernel(d: types.Data):
    efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc.worldid[efcid]
    Jaref = d.efc.Jaref[efcid]
    efc_D = d.efc.D[efcid]

    # TODO(team): active and conditionally active constraints
    active = int(Jaref < 0.0)
    d.efc.active[efcid] = active

    if active:
      # efc_force = -efc_D * Jaref * active
      d.efc.force[efcid] = -1.0 * efc_D * Jaref

      # cost = 0.5 * sum(efc_D * Jaref * Jaref * active))
      wp.atomic_add(d.efc.cost, worldid, 0.5 * efc_D * Jaref * Jaref)
    else:
      d.efc.force[efcid] = 0.0

  @kernel
  def _qfrc_constraint(d: types.Data):
    dofid, efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc.worldid[efcid]
    wp.atomic_add(
      d.qfrc_constraint[worldid],
      dofid,
      d.efc.J[efcid, dofid] * d.efc.force[efcid],
    )

  @kernel
  def _gauss(d: types.Data):
    worldid, dofid = wp.tid()
    gauss_cost = (
      0.5
      * (d.efc.Ma[worldid, dofid] - d.qfrc_smooth[worldid, dofid])
      * (d.qacc[worldid, dofid] - d.qacc_smooth[worldid, dofid])
    )
    wp.atomic_add(d.efc.gauss, worldid, gauss_cost)
    wp.atomic_add(d.efc.cost, worldid, gauss_cost)

  wp.launch(_init_cost, dim=(d.nworld), inputs=[d])

  wp.launch(_efc_kernel, dim=(d.njmax,), inputs=[d])

  # qfrc_constraint = efc_J.T @ efc_force
  d.qfrc_constraint.zero_()

  wp.launch(_qfrc_constraint, dim=(m.nv, d.njmax), inputs=[d])

  # gauss = 0.5 * (Ma - qfrc_smooth).T @ (qacc - qacc_smooth)

  wp.launch(_gauss, dim=(d.nworld, m.nv), inputs=[d])


def _update_gradient(m: types.Model, d: types.Data):
  TILE = m.nv

  @kernel
  def _grad(d: types.Data):
    worldid, dofid = wp.tid()
    grad = (
      d.efc.Ma[worldid, dofid]
      - d.qfrc_smooth[worldid, dofid]
      - d.qfrc_constraint[worldid, dofid]
    )
    d.efc.grad[worldid, dofid] = grad
    wp.atomic_add(d.efc.grad_dot, worldid, grad * grad)

  if m.opt.is_sparse:

    @kernel
    def _zero_h_lower(m: types.Model, d: types.Data):
      worldid, elementid = wp.tid()
      rowid = m.dof_tri_row[elementid]
      colid = m.dof_tri_col[elementid]
      d.efc.h[worldid, rowid, colid] = 0.0

    @kernel
    def _set_h_qM_lower_sparse(m: types.Model, d: types.Data):
      worldid, elementid = wp.tid()
      i = m.qM_fullm_i[elementid]
      j = m.qM_fullm_j[elementid]
      d.efc.h[worldid, i, j] = d.qM[worldid, 0, elementid]

  else:

    @kernel
    def _copy_lower_triangle(m: types.Model, d: types.Data):
      worldid, elementid = wp.tid()
      rowid = m.dof_tri_row[elementid]
      colid = m.dof_tri_col[elementid]
      d.efc.h[worldid, rowid, colid] = d.qM[worldid, rowid, colid]

  @kernel
  def _JTDAJ(m: types.Model, d: types.Data):
    efcid, elementid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    dofi = m.dof_tri_row[elementid]
    dofj = m.dof_tri_col[elementid]

    efc_D = d.efc.D[efcid]
    active = d.efc.active[efcid]
    if efc_D == 0.0 or active == 0:
      return

    worldid = d.efc.worldid[efcid]
    # TODO(team): sparse efc_J
    wp.atomic_add(
      d.efc.h[worldid, dofi],
      dofj,
      d.efc.J[efcid, dofi] * d.efc.J[efcid, dofj] * efc_D,
    )

  @kernel
  def _cholesky(d: types.Data):
    worldid = wp.tid()
    mat_tile = wp.tile_load(d.efc.h[worldid], shape=(TILE, TILE))
    fact_tile = wp.tile_cholesky(mat_tile)
    input_tile = wp.tile_load(d.efc.grad[worldid], shape=TILE)
    output_tile = wp.tile_cholesky_solve(fact_tile, input_tile)
    wp.tile_store(d.efc.Mgrad[worldid], output_tile)

  @kernel
  def _JTDAJ(m: types.Model, d: types.Data):
    efcid, elementid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    dofi = m.dof_tri_row[elementid]
    dofj = m.dof_tri_col[elementid]

    efc_D = d.efc.D[efcid]
    active = d.efc.active[efcid]
    if efc_D == 0.0 or active == 0:
      return

    worldid = d.efc.worldid[efcid]
    # TODO(team): sparse efc_J
    # h[worldid, dofi, dofj] += J[efcid, dofi] * J[efcid, dofj] * D[efcid]
    res = d.efc.J[efcid, dofi] * d.efc.J[efcid, dofj] * efc_D
    wp.atomic_add(d.efc.h[worldid, dofi], dofj, res)

  # grad = Ma - qfrc_smooth - qfrc_constraint
  d.efc.grad_dot.zero_()

  wp.launch(_grad, dim=(d.nworld, m.nv), inputs=[d])

  if m.opt.solver == mujoco.mjtSolver.mjSOL_CG:
    smooth.solve_m(m, d, d.efc.Mgrad, d.efc.grad)
  elif m.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON:
    # h = qM + (efc_J.T * efc_D * active) @ efc_J
    if m.opt.is_sparse:
      wp.launch(_zero_h_lower, dim=(d.nworld, m.dof_tri_row.size), inputs=[m, d])

      wp.launch(
        _set_h_qM_lower_sparse, dim=(d.nworld, m.qM_fullm_i.size), inputs=[m, d]
      )
    else:
      wp.launch(_copy_lower_triangle, dim=(d.nworld, m.dof_tri_row.size), inputs=[m, d])

    wp.launch(_JTDAJ, dim=(d.njmax, m.dof_tri_row.size), inputs=[m, d])

    wp.launch_tiled(_cholesky, dim=(d.nworld,), inputs=[d], block_dim=32)


@wp.func
def _rescale(m: types.Model, value: float) -> float:
  return value / (m.stat.meaninertia * float(wp.max(1, m.nv)))


@wp.func
def _in_bracket(x: wp.vec3, y: wp.vec3) -> bool:
  return (x[1] < y[1] and y[1] < 0.0) or (x[1] > y[1] and y[1] > 0.0)


@wp.func
def _eval_pt(quad: wp.vec3, alpha: wp.float32) -> wp.vec3:
  return wp.vec3(
    alpha * alpha * quad[2] + alpha * quad[1] + quad[0],
    2.0 * alpha * quad[2] + quad[1],
    2.0 * quad[2],
  )


@wp.func
def _safe_div(x: wp.float32, y: wp.float32) -> wp.float32:
  return x / wp.where(y != 0.0, y, types.MJ_MINVAL)


def _linesearch_iterative(m: types.Model, d: types.Data):
  @kernel
  def _gtol(m: types.Model, d: types.Data):
    # TODO(team): static m?
    worldid = wp.tid()
    snorm = wp.math.sqrt(d.efc.search_dot[worldid])
    scale = m.stat.meaninertia * wp.float(wp.max(1, m.nv))
    d.efc.gtol[worldid] = m.opt.tolerance * m.opt.ls_tolerance * snorm * scale

  @kernel
  def _jv(d: types.Data):
    efcid, dofid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    j = d.efc.J[efcid, dofid]
    search = d.efc.search[d.efc.worldid[efcid], dofid]
    wp.atomic_add(d.efc.jv, efcid, j * search)

  @kernel
  def _init_quad_gauss(m: types.Model, d: types.Data):
    worldid, dofid = wp.tid()
    search = d.efc.search[worldid, dofid]
    quad_gauss = wp.vec3()
    quad_gauss[0] = d.efc.gauss[worldid] / float(m.nv)
    quad_gauss[1] = search * (d.efc.Ma[worldid, dofid] - d.qfrc_smooth[worldid, dofid])
    quad_gauss[2] = 0.5 * search * d.efc.mv[worldid, dofid]
    wp.atomic_add(d.efc.quad_gauss, worldid, quad_gauss)

  @kernel
  def _init_quad(d: types.Data):
    efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    Jaref = d.efc.Jaref[efcid]
    jv = d.efc.jv[efcid]
    efc_D = d.efc.D[efcid]
    quad = wp.vec3()
    quad[0] = 0.5 * Jaref * Jaref * efc_D
    quad[1] = jv * Jaref * efc_D
    quad[2] = 0.5 * jv * jv * efc_D
    d.efc.quad[efcid] = quad

  @kernel
  def _init_p0_gauss(p0: wp.array(dtype=wp.vec3), d: types.Data):
    worldid = wp.tid()
    quad = d.efc.quad_gauss[worldid]
    p0[worldid] = wp.vec3(quad[0], quad[1], 2.0 * quad[2])

  @kernel
  def _init_p0(p0: wp.array(dtype=wp.vec3), d: types.Data):
    efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    # TODO(team): active and conditionally active constraints:
    if d.efc.Jaref[efcid] >= 0.0:
      return

    worldid = d.efc.worldid[efcid]
    quad = d.efc.quad[efcid]
    wp.atomic_add(p0, worldid, wp.vec3(quad[0], quad[1], 2.0 * quad[2]))

  @kernel
  def _init_lo_gauss(
    p0: wp.array(dtype=wp.vec3),
    lo: wp.array(dtype=wp.vec3),
    lo_alpha: wp.array(dtype=wp.float32),
    d: types.Data,
  ):
    worldid = wp.tid()

    pp0 = p0[worldid]
    alpha = -_safe_div(pp0[1], pp0[2])
    lo[worldid] = _eval_pt(d.efc.quad_gauss[worldid], alpha)
    lo_alpha[worldid] = alpha

  @kernel
  def _init_lo(
    lo: wp.array(dtype=wp.vec3),
    lo_alpha: wp.array(dtype=wp.float32),
    d: types.Data,
  ):
    efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc.worldid[efcid]
    alpha = lo_alpha[worldid]

    # TODO(team): active and conditionally active constraints
    if d.efc.Jaref[efcid] + alpha * d.efc.jv[efcid] < 0.0:
      wp.atomic_add(lo, worldid, _eval_pt(d.efc.quad[efcid], alpha))

  @kernel
  def _init_bounds(
    p0: wp.array(dtype=wp.vec3),
    lo: wp.array(dtype=wp.vec3),
    lo_alpha: wp.array(dtype=wp.float32),
    hi: wp.array(dtype=wp.vec3),
    hi_alpha: wp.array(dtype=wp.float32),
  ):
    worldid = wp.tid()
    pp0 = p0[worldid]
    plo = lo[worldid]
    plo_alpha = lo_alpha[worldid]
    lo_less = plo[1] < pp0[1]
    lo[worldid] = wp.where(lo_less, plo, pp0)
    lo_alpha[worldid] = wp.where(lo_less, plo_alpha, 0.0)
    hi[worldid] = wp.where(lo_less, pp0, plo)
    hi_alpha[worldid] = wp.where(lo_less, 0.0, plo_alpha)

  @kernel
  def _next_alpha_gauss(
    done: wp.array(dtype=bool),
    lo: wp.array(dtype=wp.vec3),
    lo_alpha: wp.array(dtype=wp.float32),
    hi: wp.array(dtype=wp.vec3),
    hi_alpha: wp.array(dtype=wp.float32),
    lo_next: wp.array(dtype=wp.vec3),
    lo_next_alpha: wp.array(dtype=wp.float32),
    hi_next: wp.array(dtype=wp.vec3),
    hi_next_alpha: wp.array(dtype=wp.float32),
    mid: wp.array(dtype=wp.vec3),
    mid_alpha: wp.array(dtype=wp.float32),
    d: types.Data,
  ):
    worldid = wp.tid()

    if done[worldid]:
      return

    quad = d.efc.quad_gauss[worldid]

    plo = lo[worldid]
    plo_alpha = lo_alpha[worldid]
    plo_next_alpha = plo_alpha - _safe_div(plo[1], plo[2])
    lo_next[worldid] = _eval_pt(quad, plo_next_alpha)
    lo_next_alpha[worldid] = plo_next_alpha

    phi = hi[worldid]
    phi_alpha = hi_alpha[worldid]
    phi_next_alpha = phi_alpha - _safe_div(phi[1], phi[2])
    hi_next[worldid] = _eval_pt(quad, phi_next_alpha)
    hi_next_alpha[worldid] = phi_next_alpha

    pmid_alpha = 0.5 * (plo_alpha + phi_alpha)
    mid[worldid] = _eval_pt(quad, pmid_alpha)
    mid_alpha[worldid] = pmid_alpha

  @kernel
  def _next_quad(
    done: wp.array(dtype=bool),
    lo_next: wp.array(dtype=wp.vec3),
    lo_next_alpha: wp.array(dtype=wp.float32),
    hi_next: wp.array(dtype=wp.vec3),
    hi_next_alpha: wp.array(dtype=wp.float32),
    mid: wp.array(dtype=wp.vec3),
    mid_alpha: wp.array(dtype=wp.float32),
    d: types.Data,
  ):
    efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc.worldid[efcid]

    if done[worldid]:
      return

    quad = d.efc.quad[efcid]
    jaref = d.efc.Jaref[efcid]
    jv = d.efc.jv[efcid]

    alpha = lo_next_alpha[worldid]
    # TODO(team): active and conditionally active constraints
    if jaref + alpha * jv < 0.0:
      wp.atomic_add(lo_next, worldid, _eval_pt(quad, alpha))

    alpha = hi_next_alpha[worldid]
    # TODO(team): active and conditionally active constraints
    if jaref + alpha * jv < 0.0:
      wp.atomic_add(hi_next, worldid, _eval_pt(quad, alpha))

    alpha = mid_alpha[worldid]
    # TODO(team): active and conditionally active constraints
    if jaref + alpha * jv < 0.0:
      wp.atomic_add(mid, worldid, _eval_pt(quad, alpha))

  @kernel
  def _swap(
    done: wp.array(dtype=bool),
    p0: wp.array(dtype=wp.vec3),
    lo: wp.array(dtype=wp.vec3),
    lo_alpha: wp.array(dtype=wp.float32),
    hi: wp.array(dtype=wp.vec3),
    hi_alpha: wp.array(dtype=wp.float32),
    lo_next: wp.array(dtype=wp.vec3),
    lo_next_alpha: wp.array(dtype=wp.float32),
    hi_next: wp.array(dtype=wp.vec3),
    hi_next_alpha: wp.array(dtype=wp.float32),
    mid: wp.array(dtype=wp.vec3),
    mid_alpha: wp.array(dtype=wp.float32),
    d: types.Data,
  ):
    worldid = wp.tid()

    if done[worldid]:
      return

    plo = lo[worldid]
    plo_alpha = lo_alpha[worldid]
    phi = hi[worldid]
    phi_alpha = hi_alpha[worldid]
    plo_next = lo_next[worldid]
    plo_next_alpha = lo_next_alpha[worldid]
    phi_next = hi_next[worldid]
    phi_next_alpha = hi_next_alpha[worldid]
    pmid = mid[worldid]
    pmid_alpha = mid_alpha[worldid]

    # swap lo:
    swap_lo_lo_next = _in_bracket(plo, plo_next)
    plo = wp.where(swap_lo_lo_next, plo_next, plo)
    plo_alpha = wp.where(swap_lo_lo_next, plo_next_alpha, plo_alpha)
    swap_lo_mid = _in_bracket(plo, pmid)
    plo = wp.where(swap_lo_mid, pmid, plo)
    plo_alpha = wp.where(swap_lo_mid, pmid_alpha, plo_alpha)
    swap_lo_hi_next = _in_bracket(plo, phi_next)
    plo = wp.where(swap_lo_hi_next, phi_next, plo)
    plo_alpha = wp.where(swap_lo_hi_next, phi_next_alpha, plo_alpha)
    lo[worldid] = plo
    lo_alpha[worldid] = plo_alpha
    swap_lo = swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next

    # swap hi:
    swap_hi_hi_next = _in_bracket(phi, phi_next)
    phi = wp.where(swap_hi_hi_next, phi_next, phi)
    phi_alpha = wp.where(swap_hi_hi_next, phi_next_alpha, phi_alpha)
    swap_hi_mid = _in_bracket(phi, pmid)
    phi = wp.where(swap_hi_mid, pmid, phi)
    phi_alpha = wp.where(swap_hi_mid, pmid_alpha, phi_alpha)
    swap_hi_lo_next = _in_bracket(phi, plo_next)
    phi = wp.where(swap_hi_lo_next, plo_next, phi)
    phi_alpha = wp.where(swap_hi_lo_next, plo_next_alpha, phi_alpha)
    hi[worldid] = phi
    hi_alpha[worldid] = phi_alpha
    swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next

    # if we did not adjust the interval, we are done
    # also done if either low or hi slope is nearly flat
    gtol = d.efc.gtol[worldid]
    done[worldid] = (
      (not swap_lo and not swap_hi)
      or (plo[1] < 0 and plo[1] > -gtol)
      or (phi[1] > 0 and phi[1] < gtol)
    )

    # update alpha if we have an improvement
    pp0 = p0[worldid]
    alpha = 0.0
    improved = plo[0] < pp0[0] or phi[0] < pp0[0]
    plo_better = plo[0] < phi[0]
    alpha = wp.where(improved and plo_better, plo_alpha, alpha)
    alpha = wp.where(improved and not plo_better, phi_alpha, alpha)
    d.efc.alpha[worldid] = alpha

  @kernel
  def _qacc_ma(d: types.Data):
    worldid, dofid = wp.tid()
    alpha = d.efc.alpha[worldid]
    d.qacc[worldid, dofid] += alpha * d.efc.search[worldid, dofid]
    d.efc.Ma[worldid, dofid] += alpha * d.efc.mv[worldid, dofid]

  @kernel
  def _jaref(d: types.Data):
    efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    d.efc.Jaref[efcid] += d.efc.alpha[d.efc.worldid[efcid]] * d.efc.jv[efcid]

  wp.launch(_gtol, dim=(d.nworld,), inputs=[m, d])

  # mv = qM @ search
  support.mul_m(m, d, d.efc.mv, d.efc.search)

  # jv = efc_J @ search
  # TODO(team): is there a better way of doing batched matmuls with dynamic array sizes?
  d.efc.jv.zero_()

  wp.launch(_jv, dim=(d.njmax, m.nv), inputs=[d])

  # prepare quadratics
  # quad_gauss = [gauss, search.T @ Ma - search.T @ qfrc_smooth, 0.5 * search.T @ mv]
  d.efc.quad_gauss.zero_()

  wp.launch(_init_quad_gauss, dim=(d.nworld, m.nv), inputs=[m, d])

  # quad = [0.5 * Jaref * Jaref * efc_D, jv * Jaref * efc_D, 0.5 * jv * jv * efc_D]

  wp.launch(_init_quad, dim=(d.njmax), inputs=[d])

  # linesearch points
  done = d.efc.ls_done
  done.zero_()
  p0 = d.efc.p0
  lo = d.efc.lo
  lo_alpha = d.efc.lo_alpha
  hi = d.efc.hi
  hi_alpha = d.efc.hi_alpha
  lo_next = d.efc.lo_next
  lo_next_alpha = d.efc.lo_next_alpha
  hi_next = d.efc.hi_next
  hi_next_alpha = d.efc.hi_next_alpha
  mid = d.efc.mid
  mid_alpha = d.efc.mid_alpha

  # initialize interval

  wp.launch(_init_p0_gauss, dim=(d.nworld,), inputs=[p0, d])

  wp.launch(_init_p0, dim=(d.njmax,), inputs=[p0, d])

  wp.launch(_init_lo_gauss, dim=(d.nworld,), inputs=[p0, lo, lo_alpha, d])

  wp.launch(_init_lo, dim=(d.njmax,), inputs=[lo, lo_alpha, d])

  # set the lo/hi interval bounds

  wp.launch(_init_bounds, dim=(d.nworld,), inputs=[p0, lo, lo_alpha, hi, hi_alpha])

  for _ in range(m.opt.ls_iterations):
    # note: we always launch ls_iterations kernels, but the kernels may early exit if done is true
    # this allows us to preserve cudagraph requirements (no dynamic kernel launching) at the expense
    # of extra launches
    inputs = [done, lo, lo_alpha, hi, hi_alpha, lo_next, lo_next_alpha, hi_next]
    inputs += [hi_next_alpha, mid, mid_alpha, d]
    wp.launch(_next_alpha_gauss, dim=(d.nworld,), inputs=inputs)

    inputs = [done, lo_next, lo_next_alpha, hi_next, hi_next_alpha, mid, mid_alpha]
    inputs += [d]
    wp.launch(_next_quad, dim=(d.njmax,), inputs=inputs)

    inputs = [done, p0, lo, lo_alpha, hi, hi_alpha, lo_next, lo_next_alpha, hi_next]
    inputs += [hi_next_alpha, mid, mid_alpha, d]
    wp.launch(_swap, dim=(d.nworld,), inputs=inputs)

  wp.launch(_qacc_ma, dim=(d.nworld, m.nv), inputs=[d])

  wp.launch(_jaref, dim=(d.njmax,), inputs=[d])


@event_scope
def solve(m: types.Model, d: types.Data):
  """Finds forces that satisfy constraints."""

  @kernel
  def _zero_search_dot(d: types.Data):
    worldid = wp.tid()
    d.efc.search_dot[worldid] = 0.0

  @kernel
  def _search_update(d: types.Data):
    worldid, dofid = wp.tid()
    search = -1.0 * d.efc.Mgrad[worldid, dofid]

    if wp.static(m.opt.solver == mujoco.mjtSolver.mjSOL_CG):
      search += d.efc.beta[worldid] * d.efc.search[worldid, dofid]

    d.efc.search[worldid, dofid] = search
    wp.atomic_add(d.efc.search_dot, worldid, search * search)

  @kernel
  def _done(m: types.Model, d: types.Data, solver_niter: int):
    worldid = wp.tid()
    improvement = _rescale(m, d.efc.prev_cost[worldid] - d.efc.cost[worldid])
    gradient = _rescale(m, wp.math.sqrt(d.efc.grad_dot[worldid]))
    done = solver_niter >= m.opt.iterations
    done = done or (improvement < m.opt.tolerance)
    done = done or (gradient < m.opt.tolerance)
    d.efc.done[worldid] = int(done)

  if m.opt.solver == mujoco.mjtSolver.mjSOL_CG:

    @kernel
    def _prev_grad_Mgrad(d: types.Data):
      worldid, dofid = wp.tid()
      d.efc.prev_grad[worldid, dofid] = d.efc.grad[worldid, dofid]
      d.efc.prev_Mgrad[worldid, dofid] = d.efc.Mgrad[worldid, dofid]

    @kernel
    def _zero_beta_num_den(d: types.Data):
      worldid = wp.tid()
      d.efc.beta_num[worldid] = 0.0
      d.efc.beta_den[worldid] = 0.0

    @kernel
    def _beta_num_den(d: types.Data):
      worldid, dofid = wp.tid()
      prev_Mgrad = d.efc.prev_Mgrad[worldid][dofid]
      wp.atomic_add(
        d.efc.beta_num,
        worldid,
        d.efc.grad[worldid, dofid] * (d.efc.Mgrad[worldid, dofid] - prev_Mgrad),
      )
      wp.atomic_add(
        d.efc.beta_den, worldid, d.efc.prev_grad[worldid, dofid] * prev_Mgrad
      )

    @kernel
    def _beta(d: types.Data):
      worldid = wp.tid()
      d.efc.beta[worldid] = wp.max(
        0.0, d.efc.beta_num[worldid] / wp.max(mujoco.mjMINVAL, d.efc.beta_den[worldid])
      )

  # warmstart
  kernel_copy(d.qacc, d.qacc_warmstart)

  _create_context(m, d, grad=True)

  for i in range(m.opt.iterations):
    _linesearch_iterative(m, d)

    if m.opt.solver == mujoco.mjtSolver.mjSOL_CG:
      wp.launch(_prev_grad_Mgrad, dim=(d.nworld, m.nv), inputs=[d])

    _update_constraint(m, d)
    _update_gradient(m, d)

    # polak-ribiere
    if m.opt.solver == mujoco.mjtSolver.mjSOL_CG:
      wp.launch(_zero_beta_num_den, dim=(d.nworld), inputs=[d])

      wp.launch(_beta_num_den, dim=(d.nworld, m.nv), inputs=[d])

      wp.launch(_beta, dim=(d.nworld,), inputs=[d])

    wp.launch(_zero_search_dot, dim=(d.nworld), inputs=[d])

    wp.launch(_search_update, dim=(d.nworld, m.nv), inputs=[d])

    wp.launch(_done, dim=(d.nworld,), inputs=[m, d, i])
    # TODO(team): return if all done

  kernel_copy(d.qacc_warmstart, d.qacc)
