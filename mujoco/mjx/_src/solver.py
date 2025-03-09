import warp as wp
import mujoco
from . import smooth
from . import support
from . import types
from .warp_util import event_scope
from .warp_util import kernel


@wp.struct
class Context:
  Jaref: wp.array(dtype=wp.float32, ndim=1)
  Ma: wp.array(dtype=wp.float32, ndim=2)
  grad: wp.array(dtype=wp.float32, ndim=2)
  grad_dot: wp.array(dtype=wp.float32, ndim=1)
  Mgrad: wp.array(dtype=wp.float32, ndim=2)
  search: wp.array(dtype=wp.float32, ndim=2)
  search_dot: wp.array(dtype=wp.float32, ndim=1)
  gauss: wp.array(dtype=wp.float32, ndim=1)
  cost: wp.array(dtype=wp.float32, ndim=1)
  prev_cost: wp.array(dtype=wp.float32, ndim=1)
  solver_niter: wp.array(dtype=wp.int32, ndim=1)
  active: wp.array(dtype=wp.int32, ndim=1)
  gtol: wp.array(dtype=wp.float32, ndim=1)
  mv: wp.array(dtype=wp.float32, ndim=2)
  jv: wp.array(dtype=wp.float32, ndim=1)
  quad: wp.array(dtype=wp.vec3f, ndim=1)
  quad_gauss: wp.array(dtype=wp.vec3f, ndim=1)
  h: wp.array(dtype=wp.float32, ndim=3)
  alpha: wp.array(dtype=wp.float32, ndim=1)
  prev_grad: wp.array(dtype=wp.float32, ndim=2)
  prev_Mgrad: wp.array(dtype=wp.float32, ndim=2)
  beta: wp.array(dtype=wp.float32, ndim=1)
  beta_num: wp.array(dtype=wp.float32, ndim=1)
  beta_den: wp.array(dtype=wp.float32, ndim=1)
  done: wp.array(dtype=wp.int32, ndim=1)


def _context(m: types.Model, d: types.Data) -> Context:
  ctx = Context()
  ctx.Jaref = wp.empty(shape=(d.njmax,), dtype=wp.float32)
  ctx.Ma = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.grad = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.grad_dot = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.Mgrad = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.search = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.search_dot = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.gauss = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.cost = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.prev_cost = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.solver_niter = wp.empty(shape=(d.nworld,), dtype=wp.int32)
  ctx.active = wp.empty(shape=(d.njmax,), dtype=wp.int32)
  ctx.gtol = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.mv = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.jv = wp.empty(shape=(d.njmax,), dtype=wp.float32)
  ctx.quad = wp.empty(shape=(d.njmax,), dtype=wp.vec3f)
  ctx.quad_gauss = wp.empty(shape=(d.nworld,), dtype=wp.vec3f)
  ctx.h = wp.empty(shape=(d.nworld, m.nv, m.nv), dtype=wp.float32)
  ctx.alpha = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.prev_grad = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.prev_Mgrad = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.beta = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.beta_num = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.beta_den = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.done = wp.empty(shape=(d.nworld,), dtype=wp.int32)

  return ctx


def _create_context(ctx: Context, m: types.Model, d: types.Data, grad: bool = True):
  @kernel
  def _init_context(ctx: Context):
    worldid = wp.tid()
    ctx.cost[worldid] = wp.inf
    ctx.solver_niter[worldid] = 0
    ctx.done[worldid] = 0
    if grad:
      ctx.search_dot[worldid] = 0.0

  @kernel
  def _jaref(ctx: Context, m: types.Model, d: types.Data):
    efcid, dofid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc_worldid[efcid]
    wp.atomic_add(
      ctx.Jaref,
      efcid,
      d.efc_J[efcid, dofid] * d.qacc[worldid, dofid] - d.efc_aref[efcid] / float(m.nv),
    )

  @kernel
  def _search(ctx: Context):
    worldid, dofid = wp.tid()
    search = -1.0 * ctx.Mgrad[worldid, dofid]
    ctx.search[worldid, dofid] = search
    wp.atomic_add(ctx.search_dot, worldid, search * search)

  wp.launch(_init_context, dim=(d.nworld), inputs=[ctx])

  # jaref = d.efc_J @ d.qacc - d.efc_aref
  ctx.Jaref.zero_()

  wp.launch(_jaref, dim=(d.njmax, m.nv), inputs=[ctx, m, d])

  # Ma = qM @ qacc
  support.mul_m(m, d, ctx.Ma, d.qacc)

  _update_constraint(m, d, ctx)
  if grad:
    _update_gradient(m, d, ctx)

    # search = -Mgrad
    wp.launch(_search, dim=(d.nworld, m.nv), inputs=[ctx])


def _update_constraint(m: types.Model, d: types.Data, ctx: Context):
  @kernel
  def _init_cost(ctx: Context):
    worldid = wp.tid()
    ctx.prev_cost[worldid] = ctx.cost[worldid]
    ctx.cost[worldid] = 0.0
    ctx.gauss[worldid] = 0.0

  @kernel
  def _efc_kernel(ctx: Context, d: types.Data):
    efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc_worldid[efcid]
    Jaref = ctx.Jaref[efcid]
    efc_D = d.efc_D[efcid]

    # TODO(team): active and conditionally active constraints
    active = int(Jaref < 0.0)
    ctx.active[efcid] = active

    if active:
      # efc_force = -efc_D * Jaref * active
      d.efc_force[efcid] = -1.0 * efc_D * Jaref

      # cost = 0.5 * sum(efc_D * Jaref * Jaref * active))
      wp.atomic_add(ctx.cost, worldid, 0.5 * efc_D * Jaref * Jaref)
    else:
      d.efc_force[efcid] = 0.0

  @kernel
  def _qfrc_constraint(d: types.Data):
    dofid, efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc_worldid[efcid]
    wp.atomic_add(
      d.qfrc_constraint[worldid],
      dofid,
      d.efc_J[efcid, dofid] * d.efc_force[efcid],
    )

  @kernel
  def _gauss(ctx: Context, d: types.Data):
    worldid, dofid = wp.tid()
    gauss_cost = (
      0.5
      * (ctx.Ma[worldid, dofid] - d.qfrc_smooth[worldid, dofid])
      * (d.qacc[worldid, dofid] - d.qacc_smooth[worldid, dofid])
    )
    wp.atomic_add(ctx.gauss, worldid, gauss_cost)
    wp.atomic_add(ctx.cost, worldid, gauss_cost)

  wp.launch(_init_cost, dim=(d.nworld), inputs=[ctx])

  wp.launch(_efc_kernel, dim=(d.njmax,), inputs=[ctx, d])

  # qfrc_constraint = efc_J.T @ efc_force
  d.qfrc_constraint.zero_()

  wp.launch(_qfrc_constraint, dim=(m.nv, d.njmax), inputs=[d])

  # gauss = 0.5 * (Ma - qfrc_smooth).T @ (qacc - qacc_smooth)

  wp.launch(_gauss, dim=(d.nworld, m.nv), inputs=[ctx, d])


def _update_gradient(m: types.Model, d: types.Data, ctx: Context):
  TILE = m.nv

  @kernel
  def _grad(ctx: Context, d: types.Data):
    worldid, dofid = wp.tid()
    grad = (
      ctx.Ma[worldid, dofid]
      - d.qfrc_smooth[worldid, dofid]
      - d.qfrc_constraint[worldid, dofid]
    )
    ctx.grad[worldid, dofid] = grad
    wp.atomic_add(ctx.grad_dot, worldid, grad * grad)

  if m.opt.is_sparse:

    @kernel
    def _zero_h_lower(m: types.Model, ctx: Context):
      worldid, elementid = wp.tid()
      rowid = m.dof_tri_row[elementid]
      colid = m.dof_tri_col[elementid]
      ctx.h[worldid, rowid, colid] = 0.0

    @kernel
    def _set_h_qM_lower_sparse(m: types.Model, d: types.Data, ctx: Context):
      worldid, elementid = wp.tid()
      i = m.qM_fullm_i[elementid]
      j = m.qM_fullm_j[elementid]
      ctx.h[worldid, i, j] = d.qM[worldid, 0, elementid]

  else:

    @kernel
    def _copy_lower_triangle(m: types.Model, d: types.Data, ctx: Context):
      worldid, elementid = wp.tid()
      rowid = m.dof_tri_row[elementid]
      colid = m.dof_tri_col[elementid]
      ctx.h[worldid, rowid, colid] = d.qM[worldid, rowid, colid]

  @kernel
  def _JTDAJ(ctx: Context, m: types.Model, d: types.Data):
    efcid, elementid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    dofi = m.dof_tri_row[elementid]
    dofj = m.dof_tri_col[elementid]

    efc_D = d.efc_D[efcid]
    active = ctx.active[efcid]
    if efc_D == 0.0 or active == 0:
      return

    worldid = d.efc_worldid[efcid]
    # TODO(team): sparse efc_J
    wp.atomic_add(
      ctx.h[worldid, dofi],
      dofj,
      d.efc_J[efcid, dofi] * d.efc_J[efcid, dofj] * efc_D,
    )

  @kernel(module="unique")
  def _cholesky(ctx: Context):
    worldid = wp.tid()
    mat_tile = wp.tile_load(ctx.h[worldid], shape=(TILE, TILE))
    fact_tile = wp.tile_cholesky(mat_tile)
    input_tile = wp.tile_load(ctx.grad[worldid], shape=TILE)
    output_tile = wp.tile_cholesky_solve(fact_tile, input_tile)
    wp.tile_store(ctx.Mgrad[worldid], output_tile)

  @kernel
  def _JTDAJ(ctx: Context, m: types.Model, d: types.Data):
    efcid, elementid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    dofi = m.dof_tri_row[elementid]
    dofj = m.dof_tri_col[elementid]

    efc_D = d.efc_D[efcid]
    active = ctx.active[efcid]
    if efc_D == 0.0 or active == 0:
      return

    worldid = d.efc_worldid[efcid]
    # TODO(team): sparse efc_J
    # h[worldid, dofi, dofj] += J[efcid, dofi] * J[efcid, dofj] * D[efcid]
    res = d.efc_J[efcid, dofi] * d.efc_J[efcid, dofj] * efc_D
    wp.atomic_add(ctx.h[worldid, dofi], dofj, res)

  # grad = Ma - qfrc_smooth - qfrc_constraint
  ctx.grad_dot.zero_()

  wp.launch(_grad, dim=(d.nworld, m.nv), inputs=[ctx, d])

  if m.opt.solver == mujoco.mjtSolver.mjSOL_CG:
    smooth.solve_m(m, d, ctx.grad, ctx.Mgrad)
  elif m.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON:
    # h = qM + (efc_J.T * efc_D * active) @ efc_J
    if m.opt.is_sparse:
      wp.launch(_zero_h_lower, dim=(d.nworld, m.dof_tri_row.size), inputs=[m, ctx])

      wp.launch(
        _set_h_qM_lower_sparse, dim=(d.nworld, m.qM_fullm_i.size), inputs=[m, d, ctx]
      )
    else:
      wp.launch(
        _copy_lower_triangle, dim=(d.nworld, m.dof_tri_row.size), inputs=[m, d, ctx]
      )

    wp.launch(_JTDAJ, dim=(d.njmax, m.dof_tri_row.size), inputs=[ctx, m, d])

    wp.launch_tiled(_cholesky, dim=(d.nworld,), inputs=[ctx], block_dim=32)


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
  return x / wp.select(y == 0.0, y, types.MJ_MINVAL)


@event_scope
def _linesearch_iterative(m: types.Model, d: types.Data, ctx: Context):
  @kernel
  def _gtol(m: types.Model, ctx: Context):
    # TODO(team): static m?
    worldid = wp.tid()
    snorm = wp.math.sqrt(ctx.search_dot[worldid])
    scale = m.stat.meaninertia * wp.float(wp.max(1, m.nv))
    ctx.gtol[worldid] = m.opt.tolerance * m.opt.ls_tolerance * snorm * scale

  @kernel
  def _jv(d: types.Data, ctx: Context):
    efcid, dofid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    j = d.efc_J[efcid, dofid]
    search = ctx.search[d.efc_worldid[efcid], dofid]
    wp.atomic_add(ctx.jv, efcid, j * search)

  @kernel
  def _init_quad_gauss(m: types.Model, d: types.Data, ctx: Context):
    worldid, dofid = wp.tid()
    search = ctx.search[worldid, dofid]
    quad_gauss = wp.vec3()
    quad_gauss[0] = ctx.gauss[worldid] / float(m.nv)
    quad_gauss[1] = search * (ctx.Ma[worldid, dofid] - d.qfrc_smooth[worldid, dofid])
    quad_gauss[2] = 0.5 * search * ctx.mv[worldid, dofid]
    wp.atomic_add(ctx.quad_gauss, worldid, quad_gauss)

  @kernel
  def _init_quad(d: types.Data, ctx: Context):
    efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    Jaref = ctx.Jaref[efcid]
    jv = ctx.jv[efcid]
    efc_D = d.efc_D[efcid]
    quad = wp.vec3()
    quad[0] = 0.5 * Jaref * Jaref * efc_D
    quad[1] = jv * Jaref * efc_D
    quad[2] = 0.5 * jv * jv * efc_D
    ctx.quad[efcid] = quad

  @kernel
  def _init_p0_gauss(p0: wp.array(dtype=wp.vec3), ctx: Context):
    worldid = wp.tid()
    quad = ctx.quad_gauss[worldid]
    p0[worldid] = wp.vec3(quad[0], quad[1], 2.0 * quad[2])

  @kernel
  def _init_p0(p0: wp.array(dtype=wp.vec3), d: types.Data, ctx: Context):
    efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    # TODO(team): active and conditionally active constraints:
    if ctx.Jaref[efcid] >= 0.0:
      return

    worldid = d.efc_worldid[efcid]
    quad = ctx.quad[efcid]
    wp.atomic_add(p0, worldid, wp.vec3(quad[0], quad[1], 2.0 * quad[2]))

  @kernel
  def _init_lo_gauss(
    p0: wp.array(dtype=wp.vec3),
    lo: wp.array(dtype=wp.vec3),
    lo_alpha: wp.array(dtype=wp.float32),
    ctx: Context,
  ):
    worldid = wp.tid()

    pp0 = p0[worldid]
    alpha = -_safe_div(pp0[1], pp0[2])
    lo[worldid] = _eval_pt(ctx.quad_gauss[worldid], alpha)
    lo_alpha[worldid] = alpha

  @kernel
  def _init_lo(
    lo: wp.array(dtype=wp.vec3),
    lo_alpha: wp.array(dtype=wp.float32),
    d: types.Data,
    ctx: Context,
  ):
    efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc_worldid[efcid]
    alpha = lo_alpha[worldid]

    # TODO(team): active and conditionally active constraints
    if ctx.Jaref[efcid] + alpha * ctx.jv[efcid] < 0.0:
      wp.atomic_add(lo, worldid, _eval_pt(ctx.quad[efcid], alpha))

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
    lo[worldid] = wp.select(lo_less, pp0, plo)
    lo_alpha[worldid] = wp.select(lo_less, 0.0, plo_alpha)
    hi[worldid] = wp.select(lo_less, plo, pp0)
    hi_alpha[worldid] = wp.select(lo_less, plo_alpha, 0.0)

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
    ctx: Context,
  ):
    worldid = wp.tid()

    if done[worldid]:
      return

    quad = ctx.quad_gauss[worldid]

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
    ctx: Context,
  ):
    efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc_worldid[efcid]

    if done[worldid]:
      return

    quad = ctx.quad[efcid]
    jaref = ctx.Jaref[efcid]
    jv = ctx.jv[efcid]

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
    ctx: Context,
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
    plo = wp.select(swap_lo_lo_next, plo, plo_next)
    plo_alpha = wp.select(swap_lo_lo_next, plo_alpha, plo_next_alpha)
    swap_lo_mid = _in_bracket(plo, pmid)
    plo = wp.select(swap_lo_mid, plo, pmid)
    plo_alpha = wp.select(swap_lo_mid, plo_alpha, pmid_alpha)
    swap_lo_hi_next = _in_bracket(plo, phi_next)
    plo = wp.select(swap_lo_hi_next, plo, phi_next)
    plo_alpha = wp.select(swap_lo_hi_next, plo_alpha, phi_next_alpha)
    lo[worldid] = plo
    lo_alpha[worldid] = plo_alpha
    swap_lo = swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next

    # swap hi:
    swap_hi_hi_next = _in_bracket(phi, phi_next)
    phi = wp.select(swap_hi_hi_next, phi, phi_next)
    phi_alpha = wp.select(swap_hi_hi_next, phi_alpha, phi_next_alpha)
    swap_hi_mid = _in_bracket(phi, pmid)
    phi = wp.select(swap_hi_mid, phi, pmid)
    phi_alpha = wp.select(swap_hi_mid, phi_alpha, pmid_alpha)
    swap_hi_lo_next = _in_bracket(phi, plo_next)
    phi = wp.select(swap_hi_lo_next, phi, plo_next)
    phi_alpha = wp.select(swap_hi_lo_next, phi_alpha, plo_next_alpha)
    hi[worldid] = phi
    hi_alpha[worldid] = phi_alpha
    swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next

    # if we did not adjust the interval, we are done
    # also done if either low or hi slope is nearly flat
    gtol = ctx.gtol[worldid]
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
    alpha = wp.select(improved and plo_better, alpha, plo_alpha)
    alpha = wp.select(improved and not plo_better, alpha, phi_alpha)
    ctx.alpha[worldid] = alpha

  @kernel
  def _qacc_ma(d: types.Data, ctx: Context):
    worldid, dofid = wp.tid()
    alpha = ctx.alpha[worldid]
    d.qacc[worldid, dofid] += alpha * ctx.search[worldid, dofid]
    ctx.Ma[worldid, dofid] += alpha * ctx.mv[worldid, dofid]

  @kernel
  def _jaref(d: types.Data, ctx: Context):
    efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    ctx.Jaref[efcid] += ctx.alpha[d.efc_worldid[efcid]] * ctx.jv[efcid]

  wp.launch(_gtol, dim=(d.nworld,), inputs=[m, ctx])

  # mv = qM @ search
  support.mul_m(m, d, ctx.mv, ctx.search)

  # jv = efc_J @ search
  # TODO(team): is there a better way of doing batched matmuls with dynamic array sizes?
  ctx.jv.zero_()

  wp.launch(_jv, dim=(d.njmax, m.nv), inputs=[d, ctx])

  # prepare quadratics
  # quad_gauss = [gauss, search.T @ Ma - search.T @ qfrc_smooth, 0.5 * search.T @ mv]
  ctx.quad_gauss.zero_()

  wp.launch(_init_quad_gauss, dim=(d.nworld, m.nv), inputs=[m, d, ctx])

  # quad = [0.5 * Jaref * Jaref * efc_D, jv * Jaref * efc_D, 0.5 * jv * jv * efc_D]

  wp.launch(_init_quad, dim=(d.njmax), inputs=[d, ctx])

  # linesearch points
  done = wp.zeros(shape=(d.nworld,), dtype=bool)
  p0 = wp.empty(shape=(d.nworld,), dtype=wp.vec3)
  lo = wp.empty(shape=(d.nworld,), dtype=wp.vec3)
  lo_alpha = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  hi = wp.empty(shape=(d.nworld,), dtype=wp.vec3)
  hi_alpha = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  lo_next = wp.empty(shape=(d.nworld,), dtype=wp.vec3)
  lo_next_alpha = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  hi_next = wp.empty(shape=(d.nworld,), dtype=wp.vec3)
  hi_next_alpha = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  mid = wp.empty(shape=(d.nworld,), dtype=wp.vec3)
  mid_alpha = wp.empty(shape=(d.nworld,), dtype=wp.float32)

  # initialize interval

  wp.launch(_init_p0_gauss, dim=(d.nworld,), inputs=[p0, ctx])

  wp.launch(_init_p0, dim=(d.njmax,), inputs=[p0, d, ctx])

  wp.launch(_init_lo_gauss, dim=(d.nworld,), inputs=[p0, lo, lo_alpha, ctx])

  wp.launch(_init_lo, dim=(d.njmax,), inputs=[lo, lo_alpha, d, ctx])

  # set the lo/hi interval bounds

  wp.launch(_init_bounds, dim=(d.nworld,), inputs=[p0, lo, lo_alpha, hi, hi_alpha])

  for _ in range(m.opt.ls_iterations):
    # note: we always launch ls_iterations kernels, but the kernels may early exit if done is true
    # this allows us to preserve cudagraph requirements (no dynamic kernel launching) at the expense
    # of extra launches
    inputs = [done, lo, lo_alpha, hi, hi_alpha, lo_next, lo_next_alpha, hi_next]
    inputs += [hi_next_alpha, mid, mid_alpha, ctx]
    wp.launch(_next_alpha_gauss, dim=(d.nworld,), inputs=inputs)

    inputs = [done, lo_next, lo_next_alpha, hi_next, hi_next_alpha, mid, mid_alpha]
    inputs += [d, ctx]
    wp.launch(_next_quad, dim=(d.njmax,), inputs=inputs)

    inputs = [done, p0, lo, lo_alpha, hi, hi_alpha, lo_next, lo_next_alpha, hi_next]
    inputs += [hi_next_alpha, mid, mid_alpha, ctx]
    wp.launch(_swap, dim=(d.nworld,), inputs=inputs)

  wp.launch(_qacc_ma, dim=(d.nworld, m.nv), inputs=[d, ctx])

  wp.launch(_jaref, dim=(d.njmax,), inputs=[d, ctx])


@event_scope
def solve(m: types.Model, d: types.Data):
  """Finds forces that satisfy constraints."""

  @kernel
  def _zero_search_dot(ctx: Context):
    worldid = wp.tid()
    ctx.search_dot[worldid] = 0.0

  @kernel
  def _search_update(ctx: Context):
    worldid, dofid = wp.tid()
    search = -1.0 * ctx.Mgrad[worldid, dofid]

    if wp.static(m.opt.solver == mujoco.mjtSolver.mjSOL_CG):
      search += ctx.beta[worldid] * ctx.search[worldid, dofid]

    ctx.search[worldid, dofid] = search
    wp.atomic_add(ctx.search_dot, worldid, search * search)

  @kernel
  def _done(ctx: Context, m: types.Model, solver_niter: int):
    worldid = wp.tid()
    improvement = _rescale(m, ctx.prev_cost[worldid] - ctx.cost[worldid])
    gradient = _rescale(m, wp.math.sqrt(ctx.grad_dot[worldid]))
    done = solver_niter >= m.opt.iterations
    done = done or (improvement < m.opt.tolerance)
    done = done or (gradient < m.opt.tolerance)
    ctx.done[worldid] = int(done)

  if m.opt.solver == mujoco.mjtSolver.mjSOL_CG:

    @kernel
    def _prev_grad_Mgrad(ctx: Context):
      worldid, dofid = wp.tid()
      ctx.prev_grad[worldid, dofid] = ctx.grad[worldid, dofid]
      ctx.prev_Mgrad[worldid, dofid] = ctx.Mgrad[worldid, dofid]

    @kernel
    def _zero_beta_num_den(ctx: Context):
      worldid = wp.tid()
      ctx.beta_num[worldid] = 0.0
      ctx.beta_den[worldid] = 0.0

    @kernel
    def _beta_num_den(ctx: Context):
      worldid, dofid = wp.tid()
      prev_Mgrad = ctx.prev_Mgrad[worldid][dofid]
      wp.atomic_add(
        ctx.beta_num,
        worldid,
        ctx.grad[worldid, dofid] * (ctx.Mgrad[worldid, dofid] - prev_Mgrad),
      )
      wp.atomic_add(ctx.beta_den, worldid, ctx.prev_grad[worldid, dofid] * prev_Mgrad)

    @kernel
    def _beta(ctx: Context):
      worldid = wp.tid()
      ctx.beta[worldid] = wp.max(
        0.0, ctx.beta_num[worldid] / wp.max(mujoco.mjMINVAL, ctx.beta_den[worldid])
      )

  # warmstart
  wp.copy(d.qacc, d.qacc_warmstart)

  ctx = _context(m, d)
  _create_context(ctx, m, d, grad=True)

  for i in range(m.opt.iterations):
    _linesearch_iterative(m, d, ctx)

    if m.opt.solver == mujoco.mjtSolver.mjSOL_CG:
      wp.launch(_prev_grad_Mgrad, dim=(d.nworld, m.nv), inputs=[ctx])

    _update_constraint(m, d, ctx)
    _update_gradient(m, d, ctx)

    # polak-ribiere
    if m.opt.solver == mujoco.mjtSolver.mjSOL_CG:
      wp.launch(_zero_beta_num_den, dim=(d.nworld), inputs=[ctx])

      wp.launch(_beta_num_den, dim=(d.nworld, m.nv), inputs=[ctx])

      wp.launch(_beta, dim=(d.nworld,), inputs=[ctx])

    wp.launch(_zero_search_dot, dim=(d.nworld), inputs=[ctx])

    wp.launch(_search_update, dim=(d.nworld, m.nv), inputs=[ctx])

    wp.launch(_done, dim=(d.nworld,), inputs=[ctx, m, i])
    # TODO(team): return if all done

  wp.copy(d.qacc_warmstart, d.qacc)
