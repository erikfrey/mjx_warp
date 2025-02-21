import warp as wp
import mujoco
from . import smooth
from . import support
from . import types


@wp.struct
class Context:
  Jaref: wp.array(dtype=wp.float32, ndim=1)
  Ma: wp.array(dtype=wp.float32, ndim=2)
  grad: wp.array(dtype=wp.float32, ndim=2)
  Mgrad: wp.array(dtype=wp.float32, ndim=2)
  search: wp.array(dtype=wp.float32, ndim=2)
  gauss: wp.array(dtype=wp.float32, ndim=1)
  cost: wp.array(dtype=wp.float32, ndim=1)
  prev_cost: wp.array(dtype=wp.float32, ndim=1)
  solver_niter: wp.array(dtype=wp.int32, ndim=1)
  active: wp.array(dtype=wp.int32, ndim=1)
  gtol: wp.array(dtype=wp.float32, ndim=1)
  mv: wp.array(dtype=wp.float32, ndim=2)
  jv: wp.array(dtype=wp.float32, ndim=1)
  quad: wp.array(dtype=wp.float32, ndim=2)
  quad_gauss: wp.array(dtype=wp.float32, ndim=2)
  quad_total: wp.array(dtype=wp.float32, ndim=2)
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
  ctx.Jaref = wp.empty(shape=(d.nefc_maxbatch), dtype=wp.float32)
  ctx.Ma = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.grad = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.Mgrad = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.search = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.gauss = wp.empty(shape=(d.nworld), dtype=wp.float32)
  ctx.cost = wp.empty(shape=(d.nworld), dtype=wp.float32)
  ctx.prev_cost = wp.empty(shape=(d.nworld), dtype=wp.float32)
  ctx.solver_niter = wp.empty(shape=(d.nworld), dtype=wp.int32)
  ctx.active = wp.empty(shape=(d.nefc_maxbatch), dtype=wp.int32)
  ctx.gtol = wp.empty(shape=(d.nworld), dtype=wp.float32)
  ctx.mv = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.jv = wp.empty(shape=(d.nefc_maxbatch), dtype=wp.float32)
  ctx.quad = wp.empty(shape=(d.nefc_maxbatch, 3), dtype=wp.float32)
  ctx.quad_gauss = wp.empty(shape=(d.nworld, 3), dtype=wp.float32)
  ctx.quad_total = wp.empty(shape=(d.nworld, 3), dtype=wp.float32)
  ctx.h = wp.empty(shape=(d.nworld, m.nv, m.nv), dtype=wp.float32)
  ctx.alpha = wp.empty(shape=(d.nworld), dtype=wp.float32)
  ctx.prev_grad = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.prev_Mgrad = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.beta = wp.empty(shape=(d.nworld), dtype=wp.float32)
  ctx.beta_num = wp.empty(shape=(d.nworld), dtype=wp.float32)
  ctx.beta_den = wp.empty(shape=(d.nworld), dtype=wp.float32)
  ctx.done = wp.empty(shape=(d.nworld), dtype=wp.int32)

  return ctx


def _create_context(ctx: Context, m: types.Model, d: types.Data, grad: bool = True):
  # jaref = d.efc_J @ d.qacc - d.efc_aref
  ctx.Jaref.zero_()

  @wp.kernel
  def _jaref(ctx: Context, m: types.Model, d: types.Data):
    efcid, dofid = wp.tid()
    worldid = d.efc_worldid[efcid]
    wp.atomic_add(
      ctx.Jaref,
      efcid,
      d.efc_J[efcid, dofid] * d.qacc[worldid, dofid] - d.efc_aref[efcid] / float(m.nv),
    )

  wp.launch(_jaref, dim=(d.nefc_active, m.nv), inputs=[ctx, m, d])

  # Ma = qM @ qacc
  support.mul_m(m, d, d.qacc, ctx.Ma)
  # ctx.grad.zero_()
  # ctx.Mgrad.zero_()
  # ctx.search.zero_()
  # ctx.gauss.zero_()
  ctx.cost.fill_(wp.inf)
  # ctx.prev_cost.zero_()
  ctx.solver_niter.zero_()
  # ctx.active.zero_()
  ctx.done.zero_()

  _update_constraint(m, d, ctx)
  if wp.static(grad):
    _update_gradient(m, d, ctx)

    # search = -Mgrad
    @wp.kernel
    def _search(
      Mgrad: wp.array(ndim=2, dtype=wp.float32),
      search: wp.array(ndim=2, dtype=wp.float32),
    ):
      worldid, dofid = wp.tid()
      search[worldid, dofid] = -1.0 * Mgrad[worldid, dofid]

    wp.launch(_search, dim=(d.nworld, m.nv), inputs=[ctx.Mgrad, ctx.search])


@wp.struct
class LSPoint:
  alpha: wp.array(dtype=wp.float32, ndim=1)
  cost: wp.array(dtype=wp.float32, ndim=1)
  deriv_0: wp.array(dtype=wp.float32, ndim=1)
  deriv_1: wp.array(dtype=wp.float32, ndim=1)


def _lspoint(d: types.Data) -> LSPoint:
  ls_pnt = LSPoint()
  ls_pnt.alpha = wp.empty(shape=(d.nworld), dtype=wp.float32)
  ls_pnt.cost = wp.empty(shape=(d.nworld), dtype=wp.float32)
  ls_pnt.deriv_0 = wp.empty(shape=(d.nworld), dtype=wp.float32)
  ls_pnt.deriv_1 = wp.empty(shape=(d.nworld), dtype=wp.float32)

  return ls_pnt


def _create_lspoint(ls_pnt: LSPoint, m: types.Model, d: types.Data, ctx: Context):
  wp.copy(ctx.quad_total, ctx.quad_gauss)

  @wp.kernel
  def _quad(ctx: Context, d: types.Data):
    efcid = wp.tid()
    worldid = d.efc_worldid[efcid]
    x = ctx.Jaref[efcid] + ctx.alpha[worldid] * ctx.jv[efcid]
    # TODO(team): active and conditionally active constraints
    active = float(x < 0.0)
    wp.atomic_add(ctx.quad_total[worldid], 0, ctx.quad[efcid, 0] * active)
    wp.atomic_add(ctx.quad_total[worldid], 1, ctx.quad[efcid, 1] * active)
    wp.atomic_add(ctx.quad_total[worldid], 2, ctx.quad[efcid, 2] * active)

  wp.launch(_quad, dim=(d.nefc_active), inputs=[ctx, d])

  @wp.kernel
  def _cost_deriv01(ls_pnt: LSPoint, ctx: Context):
    worldid = wp.tid()
    alpha = ls_pnt.alpha[worldid]
    alpha_sq = alpha * alpha
    quad_total0 = ctx.quad_total[worldid, 0]
    quad_total1 = ctx.quad_total[worldid, 1]
    quad_total2 = ctx.quad_total[worldid, 2]

    ls_pnt.cost[worldid] = alpha_sq * quad_total2 + alpha * quad_total1 + quad_total0
    ls_pnt.deriv_0[worldid] = 2.0 * alpha * quad_total2 + quad_total1
    ls_pnt.deriv_1[worldid] = 2.0 * quad_total2 + float(quad_total2 == 0.0)

  wp.launch(_cost_deriv01, dim=(d.nworld), inputs=[ls_pnt, ctx])


@wp.struct
class LSContext:
  p0: LSPoint
  lo: LSPoint
  lo_next: LSPoint
  hi: LSPoint
  hi_next: LSPoint
  mid: LSPoint
  swap: wp.array(ndim=1, dtype=wp.int32)
  ls_iter: wp.array(ndim=1, dtype=wp.int32)
  done: wp.array(ndim=1, dtype=wp.int32)


def _create_lscontext(m: types.Model, d: types.Data, ctx: Context) -> LSContext:
  ls_ctx = LSContext()

  ls_ctx.p0 = _lspoint(d)
  ls_ctx.lo = _lspoint(d)
  ls_ctx.lo_next = _lspoint(d)
  ls_ctx.hi = _lspoint(d)
  ls_ctx.hi_next = _lspoint(d)
  ls_ctx.mid = _lspoint(d)

  ls_ctx.swap = wp.empty(shape=(d.nworld), dtype=wp.int32)
  ls_ctx.ls_iter = wp.empty(shape=(d.nworld), dtype=wp.int32)
  ls_ctx.done = wp.zeros((d.nworld), dtype=wp.int32)

  return ls_ctx


def _update_constraint(m: types.Model, d: types.Data, ctx: Context):
  wp.copy(ctx.prev_cost, ctx.cost)
  ctx.cost.zero_()

  @wp.kernel
  def _efc_kernel(ctx: Context, d: types.Data):
    efcid = wp.tid()
    worldid = d.efc_worldid[efcid]
    Jaref = ctx.Jaref[efcid]
    efc_D = d.efc_D[efcid]

    # TODO(team): active and conditionally active constraints
    active = int(Jaref < 0.0)
    ctx.active[efcid] = active

    # efc_force = -efc_D * Jaref * active
    d.efc_force[efcid] = -1.0 * efc_D * Jaref * float(active)

    # cost = 0.5 * sum(efc_D * Jaref * Jaref * active))
    wp.atomic_add(ctx.cost, worldid, 0.5 * efc_D * Jaref * Jaref * float(active))

  wp.launch(_efc_kernel, dim=(d.nefc_active), inputs=[ctx, d])

  # qfrc_constraint = efc_J.T @ efc_force
  d.qfrc_constraint.zero_()

  @wp.kernel
  def _qfrc_constraint(d: types.Data):
    dofid, efcid = wp.tid()
    worldid = d.efc_worldid[efcid]
    wp.atomic_add(
      d.qfrc_constraint[worldid],
      dofid,
      d.efc_J[efcid, dofid] * d.efc_force[efcid],
    )

  wp.launch(_qfrc_constraint, dim=(m.nv, d.nefc_active), inputs=[d])

  # gauss = 0.5 * (Ma - qfrc_smooth).T @ (qacc - qacc_smooth)
  ctx.gauss.zero_()

  @wp.kernel
  def _gauss(ctx: Context, d: types.Data):
    worldid, dofid = wp.tid()
    gauss_cost = (
      0.5
      * (ctx.Ma[worldid, dofid] - d.qfrc_smooth[worldid, dofid])
      * (d.qacc[worldid, dofid] - d.qacc_smooth[worldid, dofid])
    )
    wp.atomic_add(ctx.gauss, worldid, gauss_cost)
    wp.atomic_add(ctx.cost, worldid, gauss_cost)

  wp.launch(_gauss, dim=(d.nworld, m.nv), inputs=[ctx, d])


def _update_gradient(m: types.Model, d: types.Data, ctx: Context):
  # grad = Ma - qfrc_smooth - qfrc_constraint
  @wp.kernel
  def _grad(ctx: Context, d: types.Data):
    worldid, dofid = wp.tid()
    ctx.grad[worldid, dofid] = (
      ctx.Ma[worldid, dofid]
      - d.qfrc_smooth[worldid, dofid]
      - d.qfrc_constraint[worldid, dofid]
    )

  wp.launch(_grad, dim=(d.nworld, m.nv), inputs=[ctx, d])

  if m.opt.solver == 1:  # CG
    smooth.solve_m(m, d, ctx.grad, ctx.Mgrad)
  elif m.opt.solver == 2:  # Newton
    # TODO(team): sparse version
    # h = qM + (efc_J.T * efc_D * active) @ efc_J
    wp.copy(ctx.h, d.qM)

    @wp.kernel
    def _JTDAJ(ctx: Context, d: types.Data):
      dofi, dofj, efcid = wp.tid()
      worldid = d.efc_worldid[efcid]
      wp.atomic_add(
        ctx.h[worldid, dofi],
        dofj,
        d.efc_J[efcid, dofi]
        * d.efc_J[efcid, dofj]
        * d.efc_D[efcid]
        * float(ctx.active[efcid]),
      )

    wp.launch(_JTDAJ, dim=(m.nv, m.nv, d.nefc_active), inputs=[ctx, d])

    TILE = m.nv

    @wp.kernel
    def _cholesky(ctx: Context):
      worldid = wp.tid()
      mat_tile = wp.tile_load(ctx.h[worldid], shape=(TILE, TILE))
      fact_tile = wp.tile_cholesky(mat_tile)
      input_tile = wp.tile_load(ctx.grad[worldid], shape=TILE)
      output_tile = wp.tile_cholesky_solve(fact_tile, input_tile)
      wp.tile_store(ctx.Mgrad[worldid], output_tile)

    wp.launch_tiled(_cholesky, dim=(d.nworld), inputs=[ctx], block_dim=256)


@wp.func
def _rescale(m: types.Model, value: float) -> float:
  return value / (m.stat.meaninertia * float(wp.max(1, m.nv)))


@wp.func
def _in_bracket(x: float, y: float) -> bool:
  return (x < y) and (y < 0.0) or (x > y) and (y > 0.0)


def _linesearch(m: types.Model, d: types.Data, ctx: Context):
  NV = m.nv

  @wp.kernel
  def _gtol(ctx: Context, m: types.Model):
    worldid = wp.tid()
    sum = float(0.0)
    for i in range(NV):
      search = ctx.search[worldid, i]
      sum += search * search
    smag = wp.math.sqrt(sum) * m.stat.meaninertia * float(wp.max(1, NV))
    ctx.gtol[worldid] = m.opt.tolerance * m.opt.ls_tolerance * smag

  wp.launch(_gtol, dim=(d.nworld), inputs=[ctx, m])

  # mv = qM @ search
  support.mul_m(m, d, ctx.search, ctx.mv)

  # jv = efc_J @ search
  ctx.jv.zero_()

  @wp.kernel
  def _jv(ctx: Context, d: types.Data):
    efcid, dofid = wp.tid()
    worldid = d.efc_worldid[efcid]
    wp.atomic_add(
      ctx.jv,
      efcid,
      d.efc_J[efcid, dofid] * ctx.search[worldid, dofid],
    )

  wp.launch(_jv, dim=(d.nefc_active, m.nv), inputs=[ctx, d])

  # prepare quadratics
  # quad_gauss = [gauss, search.T @ Ma - search.T @ qfrc_smooth, 0.5 * search.T @ mv]
  @wp.kernel
  def _quad_gauss0(ctx: Context):
    worldid = wp.tid()
    ctx.quad_gauss[worldid, 0] = ctx.gauss[worldid]
    ctx.quad_gauss[worldid, 1] = 0.0
    ctx.quad_gauss[worldid, 2] = 0.0

  wp.launch(_quad_gauss0, dim=(d.nworld), inputs=[ctx])

  @wp.kernel
  def _quad_gauss12(ctx: Context, d: types.Data):
    worldid, dofid = wp.tid()
    search = ctx.search[worldid, dofid]
    wp.atomic_add(
      ctx.quad_gauss[worldid],
      1,
      search * (ctx.Ma[worldid, dofid] - d.qfrc_smooth[worldid, dofid]),
    )
    wp.atomic_add(ctx.quad_gauss[worldid], 2, 0.5 * search * ctx.mv[worldid, dofid])

  wp.launch(_quad_gauss12, dim=(d.nworld, m.nv), inputs=[ctx, d])

  # quad = [0.5 * Jaref * Jaref * efc_D, jv * Jaref * efc_D, 0.5 * jv * jv * efc_D]
  @wp.kernel
  def _quad(ctx: Context, d: types.Data):
    efcid = wp.tid()
    Jaref = ctx.Jaref[efcid]
    jv = ctx.jv[efcid]
    efc_D = d.efc_D[efcid]
    ctx.quad[efcid, 0] = 0.5 * Jaref * Jaref * efc_D
    ctx.quad[efcid, 1] = jv * Jaref * efc_D
    ctx.quad[efcid, 2] = 0.5 * jv * jv * efc_D

  wp.launch(_quad, dim=(d.nefc_active), inputs=[ctx, d])

  # initialize interval
  ls_ctx = _create_lscontext(m, d, ctx)

  ls_ctx.p0.alpha.zero_()
  _create_lspoint(ls_ctx.p0, m, d, ctx)

  @wp.kernel
  def _lo_alpha(lo: LSPoint, p0: LSPoint, ctx: Context):
    worldid = wp.tid()
    lo.alpha[worldid] = p0.alpha[worldid] - p0.deriv_0[worldid] / p0.deriv_1[worldid]

  wp.launch(_lo_alpha, dim=(d.nworld), inputs=[ls_ctx.lo, ls_ctx.p0, ctx])

  _create_lspoint(ls_ctx.lo, m, d, ctx)

  @wp.kernel
  def _tree_map(ls_ctx: LSContext):
    worldid = wp.tid()

    lesser = float(ls_ctx.lo.deriv_0[worldid] < ls_ctx.p0.deriv_0[worldid])
    not_lesser = 1.0 - lesser

    ls_ctx.hi.alpha[worldid] = (
      lesser * ls_ctx.p0.alpha[worldid] + not_lesser * ls_ctx.lo.alpha[worldid]
    )
    ls_ctx.hi.cost[worldid] = (
      lesser * ls_ctx.p0.cost[worldid] + not_lesser * ls_ctx.lo.cost[worldid]
    )
    ls_ctx.hi.deriv_0[worldid] = (
      lesser * ls_ctx.p0.deriv_0[worldid] + not_lesser * ls_ctx.lo.deriv_0[worldid]
    )
    ls_ctx.hi.deriv_1[worldid] = (
      lesser * ls_ctx.p0.deriv_1[worldid] + not_lesser * ls_ctx.lo.deriv_1[worldid]
    )

    ls_ctx.lo.alpha[worldid] = (
      lesser * ls_ctx.lo.alpha[worldid] + not_lesser * ls_ctx.p0.alpha[worldid]
    )
    ls_ctx.lo.cost[worldid] = (
      lesser * ls_ctx.lo.cost[worldid] + not_lesser * ls_ctx.p0.cost[worldid]
    )
    ls_ctx.lo.deriv_0[worldid] = (
      lesser * ls_ctx.lo.deriv_0[worldid] + not_lesser * ls_ctx.p0.deriv_0[worldid]
    )
    ls_ctx.lo.deriv_1[worldid] = (
      lesser * ls_ctx.lo.deriv_1[worldid] + not_lesser * ls_ctx.p0.deriv_1[worldid]
    )

  wp.launch(_tree_map, dim=(d.nworld), inputs=[ls_ctx])

  ls_ctx.swap.fill_(1)
  ls_ctx.ls_iter.fill_(0)

  for i in range(m.opt.ls_iterations):

    @wp.kernel
    def _alpha_lo_next_hi_next_mid(ls_ctx: LSContext):
      worldid = wp.tid()
      ls_ctx.lo_next.alpha[worldid] = (
        ls_ctx.lo.alpha[worldid]
        - ls_ctx.lo.deriv_0[worldid] / ls_ctx.lo.deriv_1[worldid]
      )
      ls_ctx.hi_next.alpha[worldid] = (
        ls_ctx.hi.alpha[worldid]
        - ls_ctx.hi.deriv_0[worldid] / ls_ctx.hi.deriv_1[worldid]
      )
      ls_ctx.mid.alpha[worldid] = 0.5 * (
        ls_ctx.lo.alpha[worldid] + ls_ctx.hi.alpha[worldid]
      )

    wp.launch(_alpha_lo_next_hi_next_mid, dim=(d.nworld), inputs=[ls_ctx])

    _create_lspoint(ls_ctx.lo_next, m, d, ctx)
    _create_lspoint(ls_ctx.hi_next, m, d, ctx)
    _create_lspoint(ls_ctx.mid, m, d, ctx)

    @wp.kernel
    def _swap_lo_hi(ls_ctx: LSContext):
      worldid = wp.tid()

      ls_ctx.ls_iter[worldid] += 1

      lo_alpha = ls_ctx.lo.alpha[worldid]
      lo_cost = ls_ctx.lo.cost[worldid]
      lo_deriv_0 = ls_ctx.lo.deriv_0[worldid]
      lo_deriv_1 = ls_ctx.lo.deriv_1[worldid]

      lo_next_alpha = ls_ctx.lo_next.alpha[worldid]
      lo_next_cost = ls_ctx.lo_next.cost[worldid]
      lo_next_deriv_0 = ls_ctx.lo_next.deriv_0[worldid]
      lo_next_deriv_1 = ls_ctx.lo_next.deriv_1[worldid]

      hi_alpha = ls_ctx.hi.alpha[worldid]
      hi_cost = ls_ctx.hi.cost[worldid]
      hi_deriv_0 = ls_ctx.hi.deriv_0[worldid]
      hi_deriv_1 = ls_ctx.hi.deriv_1[worldid]

      hi_next_alpha = ls_ctx.hi_next.alpha[worldid]
      hi_next_cost = ls_ctx.hi_next.cost[worldid]
      hi_next_deriv_0 = ls_ctx.hi_next.deriv_0[worldid]
      hi_next_deriv_1 = ls_ctx.hi_next.deriv_1[worldid]

      mid_alpha = ls_ctx.mid.alpha[worldid]
      mid_cost = ls_ctx.mid.cost[worldid]
      mid_deriv_0 = ls_ctx.mid.deriv_0[worldid]
      mid_deriv_1 = ls_ctx.mid.deriv_1[worldid]

      swap_lo_next = _in_bracket(lo_deriv_0, lo_next_deriv_0)
      lo_alpha = (
        float(swap_lo_next) * lo_next_alpha + (1.0 - float(swap_lo_next)) * lo_alpha
      )
      lo_cost = (
        float(swap_lo_next) * lo_next_cost + (1.0 - float(swap_lo_next)) * lo_cost
      )
      lo_deriv_0 = (
        float(swap_lo_next) * lo_next_deriv_0 + (1.0 - float(swap_lo_next)) * lo_deriv_0
      )
      lo_deriv_1 = (
        float(swap_lo_next) * lo_next_deriv_1 + (1.0 - float(swap_lo_next)) * lo_deriv_1
      )

      swap_lo_mid = _in_bracket(lo_deriv_0, mid_deriv_0)
      lo_alpha = float(swap_lo_mid) * mid_alpha + (1.0 - float(swap_lo_mid)) * lo_alpha
      lo_cost = float(swap_lo_mid) * mid_cost + (1.0 - float(swap_lo_mid)) * lo_cost
      lo_deriv_0 = (
        float(swap_lo_mid) * mid_deriv_0 + (1.0 - float(swap_lo_mid)) * lo_deriv_0
      )
      lo_deriv_1 = (
        float(swap_lo_mid) * mid_deriv_1 + (1.0 - float(swap_lo_mid)) * lo_deriv_1
      )

      swap_lo_hi_next = _in_bracket(lo_deriv_0, hi_next_deriv_0)
      lo_alpha = (
        float(swap_lo_hi_next) * hi_next_alpha
        + (1.0 - float(swap_lo_hi_next)) * lo_alpha
      )
      lo_cost = (
        float(swap_lo_hi_next) * hi_next_cost + (1.0 - float(swap_lo_hi_next)) * lo_cost
      )
      lo_deriv_0 = (
        float(swap_lo_hi_next) * hi_next_deriv_0
        + (1.0 - float(swap_lo_hi_next)) * lo_deriv_0
      )
      lo_deriv_1 = (
        float(swap_lo_hi_next) * hi_next_deriv_1
        + (1.0 - float(swap_lo_hi_next)) * lo_deriv_1
      )

      swap_hi_next = _in_bracket(hi_deriv_0, hi_next_deriv_0)
      hi_alpha = (
        float(swap_hi_next) * hi_next_alpha + (1.0 - float(swap_hi_next)) * hi_alpha
      )
      hi_cost = (
        float(swap_hi_next) * hi_next_cost + (1.0 - float(swap_hi_next)) * hi_cost
      )
      hi_deriv_0 = (
        float(swap_hi_next) * hi_next_deriv_0 + (1.0 - float(swap_hi_next)) * hi_deriv_0
      )
      hi_deriv_1 = (
        float(swap_hi_next) * hi_next_deriv_1 + (1.0 - float(swap_hi_next)) * hi_deriv_1
      )

      swap_hi_mid = _in_bracket(hi_deriv_0, mid_deriv_0)
      hi_alpha = float(swap_hi_mid) * mid_alpha + (1.0 - float(swap_hi_mid)) * hi_alpha
      hi_cost = float(swap_hi_mid) * mid_cost + (1.0 - float(swap_hi_mid)) * hi_cost
      hi_deriv_0 = (
        float(swap_hi_mid) * mid_deriv_0 + (1.0 - float(swap_hi_mid)) * hi_deriv_0
      )
      hi_deriv_1 = (
        float(swap_hi_mid) * mid_deriv_1 + (1.0 - float(swap_hi_mid)) * hi_deriv_1
      )

      swap_hi_lo_next = _in_bracket(hi_deriv_0, lo_next_deriv_0)
      hi_alpha = (
        float(swap_hi_lo_next) * lo_next_alpha
        + (1.0 - float(swap_hi_lo_next)) * hi_alpha
      )
      hi_cost = (
        float(swap_hi_lo_next) * lo_next_cost + (1.0 - float(swap_hi_lo_next)) * hi_cost
      )
      hi_deriv_0 = (
        float(swap_hi_lo_next) * lo_next_deriv_0
        + (1.0 - float(swap_hi_lo_next)) * hi_deriv_0
      )
      hi_deriv_1 = (
        float(swap_hi_lo_next) * lo_next_deriv_1
        + (1.0 - float(swap_hi_lo_next)) * hi_deriv_1
      )

      ls_ctx.lo.alpha[worldid] = lo_alpha
      ls_ctx.lo.cost[worldid] = lo_cost
      ls_ctx.lo.deriv_0[worldid] = lo_deriv_0
      ls_ctx.lo.deriv_1[worldid] = lo_deriv_1

      ls_ctx.hi.alpha[worldid] = hi_alpha
      ls_ctx.hi.cost[worldid] = hi_cost
      ls_ctx.hi.deriv_0[worldid] = hi_deriv_0
      ls_ctx.hi.deriv_1[worldid] = hi_deriv_1

      swap = swap_lo_next or swap_lo_mid or swap_lo_hi_next
      swap = swap or swap_hi_next or swap_hi_mid or swap_hi_lo_next
      ls_ctx.swap[worldid] = int(swap)

    wp.launch(_swap_lo_hi, dim=(d.nworld), inputs=[ls_ctx])

    @wp.kernel
    def _done(ls_ctx: LSContext, ctx: Context, m: types.Model, ls_iter: int):
      worldid = wp.tid()
      done = ls_iter >= m.opt.ls_iterations
      done = done or (1 - ls_ctx.swap[worldid])
      done = done or (
        (ls_ctx.lo.deriv_0[worldid] < 0.0)
        and (ls_ctx.lo.deriv_0[worldid] > -ctx.gtol[worldid])
      )
      done = done or (
        (ls_ctx.hi.deriv_0[worldid] > 0.0)
        and (ls_ctx.hi.deriv_0[worldid] < ctx.gtol[worldid])
      )
      ls_ctx.done[worldid] = int(done)

    wp.launch(_done, dim=(d.nworld), inputs=[ls_ctx, ctx, m, i])
    # TODO(team): return if all done

  @wp.kernel
  def _alpha(ctx: Context, ls_ctx: LSContext):
    worldid = wp.tid()
    p0_cost = ls_ctx.p0.cost[worldid]
    lo_cost = ls_ctx.lo.cost[worldid]
    hi_cost = ls_ctx.hi.cost[worldid]

    improvement = float((lo_cost < p0_cost) or (hi_cost < p0_cost))
    lo_hi_cost = float(lo_cost < hi_cost)
    ctx.alpha[worldid] = improvement * (
      lo_hi_cost * ls_ctx.lo.alpha[worldid]
      + (1.0 - lo_hi_cost) * ls_ctx.hi.alpha[worldid]
    )

  wp.launch(_alpha, dim=(d.nworld), inputs=[ctx, ls_ctx])

  @wp.kernel
  def _qacc_ma(ctx: Context, d: types.Data):
    worldid, dofid = wp.tid()
    alpha = ctx.alpha[worldid]
    d.qacc[worldid, dofid] += alpha * ctx.search[worldid, dofid]
    ctx.Ma[worldid, dofid] += alpha * ctx.mv[worldid, dofid]

  wp.launch(_qacc_ma, dim=(d.nworld, m.nv), inputs=[ctx, d])

  @wp.kernel
  def _jaref(ctx: Context, d: types.Data):
    efcid = wp.tid()
    worldid = d.efc_worldid[efcid]
    ctx.Jaref[efcid] += ctx.alpha[worldid] * ctx.jv[efcid]

  wp.launch(_jaref, dim=(d.nefc_active), inputs=[ctx, d])


def solve(m: types.Model, d: types.Data):
  """Finds forces that satisfy constraints."""

  # warmstart
  wp.copy(d.qacc, d.qacc_warmstart)

  ctx = _context(m, d)
  _create_context(ctx, m, d, grad=True)

  for i in range(m.opt.iterations):
    _linesearch(m, d, ctx)
    wp.copy(ctx.prev_grad, ctx.grad)
    wp.copy(ctx.prev_Mgrad, ctx.Mgrad)
    _update_constraint(m, d, ctx)
    _update_gradient(m, d, ctx)

    if m.opt.solver == 2:  # Newton

      @wp.kernel
      def _search_newton(ctx: Context):
        worldid, dofid = wp.tid()
        ctx.search[worldid, dofid] = -1.0 * ctx.Mgrad[worldid, dofid]

      wp.launch(_search_newton, dim=(d.nworld, m.nv), inputs=[ctx])
    else:  # polak-ribiere
      ctx.beta_num.zero_()
      ctx.beta_den.zero_()

      @wp.kernel
      def _beta_num_den(ctx: Context):
        worldid, dofid = wp.tid()
        prev_Mgrad = ctx.prev_Mgrad[worldid][dofid]
        wp.atomic_add(
          ctx.beta_num,
          worldid,
          ctx.grad[worldid, dofid] * (ctx.Mgrad[worldid, dofid] - prev_Mgrad),
        )
        wp.atomic_add(ctx.beta_den, worldid, ctx.prev_grad[worldid, dofid] * prev_Mgrad)

      wp.launch(_beta_num_den, dim=(d.nworld, m.nv), inputs=[ctx])

      @wp.kernel
      def _beta(ctx: Context):
        worldid = wp.tid()
        ctx.beta[worldid] = wp.max(
          0.0, ctx.beta_num[worldid] / wp.max(mujoco.mjMINVAL, ctx.beta_den[worldid])
        )

      wp.launch(_beta, dim=(d.nworld), inputs=[ctx])

      @wp.kernel
      def _search_cg(ctx: Context):
        worldid, dofid = wp.tid()
        ctx.search[worldid, dofid] = (
          -1.0 * ctx.Mgrad[worldid, dofid]
          + ctx.beta[worldid] * ctx.search[worldid, dofid]
        )

      wp.launch(_search_cg, dim=(d.nworld, m.nv), inputs=[ctx])

    NV = m.nv

    @wp.kernel
    def _done(ctx: Context, m: types.Model, solver_niter: int):
      worldid = wp.tid()
      improvement = _rescale(m, ctx.prev_cost[worldid] - ctx.cost[worldid])
      sum = float(0.0)
      for i in range(NV):
        grad = ctx.grad[worldid, i]
        sum += grad * grad
      gradient = _rescale(m, wp.math.sqrt(sum))
      done = solver_niter >= m.opt.iterations
      done = done or (improvement < m.opt.tolerance)
      done = done or (gradient < m.opt.tolerance)
      ctx.done[worldid] = int(done)

    wp.launch(_done, dim=(d.nworld), inputs=[ctx, m, i])
    # TODO(team): return if all done

  wp.copy(d.qacc_warmstart, d.qacc)
