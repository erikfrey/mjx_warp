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

import warp as wp

from .warp_util import event_scope
from .types import Data
from .types import DisableBit
from .types import MJ_MINVAL
from .types import Model
from .types import vec5
from .support import where
from .support import group_key


@wp.func
def _geom_filter(m: Model, geom1: int, geom2: int, filterparent: bool) -> bool:
  bodyid1 = m.geom_bodyid[geom1]
  bodyid2 = m.geom_bodyid[geom2]
  contype1 = m.geom_contype[geom1]
  contype2 = m.geom_contype[geom2]
  conaffinity1 = m.geom_conaffinity[geom1]
  conaffinity2 = m.geom_conaffinity[geom2]
  weldid1 = m.body_weldid[bodyid1]
  weldid2 = m.body_weldid[bodyid2]
  weld_parentid1 = m.body_weldid[m.body_parentid[weldid1]]
  weld_parentid2 = m.body_weldid[m.body_parentid[weldid2]]

  self_collision = weldid1 == weldid2
  parent_child_collision = (
    filterparent
    and (weldid1 != 0)
    and (weldid2 != 0)
    and ((weldid1 == weld_parentid2) or (weldid2 == weld_parentid1))
  )
  mask = (contype1 and conaffinity2) or (contype2 and conaffinity1)

  return mask and (not self_collision) and (not parent_child_collision)


@wp.func
def _geom_pair(m: Model, geom1: int, geom2: int) -> wp.vec2i:
  if m.geom_type[geom1] > m.geom_type[geom2]:
    return wp.vec2i(geom2, geom1)
  else:
    return wp.vec2i(geom1, geom2)


@wp.struct
class AABB:
  min: wp.vec3
  max: wp.vec3


@wp.func
def transform_aabb(
  aabb_pos: wp.vec3, aabb_size: wp.vec3, pos: wp.vec3, ori: wp.mat33
) -> AABB:
  aabb = AABB()
  aabb.max = wp.vec3(-1000000000.0, -1000000000.0, -1000000000.0)
  aabb.min = wp.vec3(1000000000.0, 1000000000.0, 1000000000.0)

  for i in range(8):
    corner = wp.vec3(aabb_size.x, aabb_size.y, aabb_size.z)
    if i % 2 == 0:
      corner.x = -corner.x
    if (i // 2) % 2 == 0:
      corner.y = -corner.y
    if i < 4:
      corner.z = -corner.z
    corner_world = ori * (corner + aabb_pos) + pos
    aabb.max = wp.max(aabb.max, corner_world)
    aabb.min = wp.min(aabb.min, corner_world)

  return aabb


@wp.kernel
def get_dyn_geom_aabb(
  m: Model,
  d: Data,
):
  env_id, gid = wp.tid()

  pos = d.geom_xpos[env_id, gid]
  ori = d.geom_xmat[env_id, gid]

  aabb_pos = m.geom_aabb[gid, 0]
  aabb_size = m.geom_aabb[gid, 1]

  aabb = transform_aabb(aabb_pos, aabb_size, pos, ori)

  # Write results to output
  d.dyn_geom_aabb[env_id, gid, 0] = aabb.min
  d.dyn_geom_aabb[env_id, gid, 1] = aabb.max


@wp.func
def overlap(
  world_id: int,
  a: int,
  b: int,
  boxes: wp.array(dtype=wp.vec3, ndim=3),
) -> bool:
  # Extract centers and sizes
  a_min = boxes[world_id, a, 0]
  a_max = boxes[world_id, a, 1]
  b_min = boxes[world_id, b, 0]
  b_max = boxes[world_id, b, 1]

  return not (
    a_min.x > b_max.x
    or b_min.x > a_max.x
    or a_min.y > b_max.y
    or b_min.y > a_max.y
    or a_min.z > b_max.z
    or b_min.z > a_max.z
  )


@wp.kernel
def broadphase_project_boxes_onto_sweep_direction_kernel(
  d: Data,
):
  worldId, i = wp.tid()

  box_min = d.dyn_geom_aabb[worldId, i, 0]
  box_max = d.dyn_geom_aabb[worldId, i, 1]
  c = (box_min + box_max) * 0.5
  box_half_size = (box_max - box_min) * 0.5

  # Use fixed direction vector and its absolute values
  direction = wp.vec3(0.5935, 0.7790, 0.1235)
  direction = wp.normalize(direction)
  abs_dir = wp.vec3(abs(direction.x), abs(direction.y), abs(direction.z))

  center = wp.dot(direction, c)
  d_val = wp.dot(box_half_size, abs_dir)
  f = center - d_val

  # Store results in the data arrays
  d.box_projections_lower[worldId, i] = f
  d.box_projections_upper[worldId, i] = center + d_val
  d.box_sorting_indexer[worldId, i] = i

  if i == 0:
    d.broadphase_result_count[worldId] = 0  # Initialize result count to 0


@wp.kernel
def reorder_bounding_boxes_kernel(
  d: Data,
):
  worldId, i = wp.tid()

  # Get the index from the data indexer
  mapped = d.box_sorting_indexer[worldId, i]

  # Get the box from the original boxes array
  box_min = d.dyn_geom_aabb[worldId, mapped, 0]
  box_max = d.dyn_geom_aabb[worldId, mapped, 1]

  # Reorder the box into the sorted array
  d.boxes_sorted[worldId, i, 0] = box_min
  d.boxes_sorted[worldId, i, 1] = box_max


@wp.func
def find_first_greater_than(
  worldId: int,
  starts: wp.array(dtype=wp.float32, ndim=2),
  value: wp.float32,
  low: int,
  high: int,
) -> int:
  while low < high:
    mid = (low + high) >> 1
    if starts[worldId, mid] > value:
      high = mid
    else:
      low = mid + 1
  return low


@wp.kernel
def broadphase_sweep_and_prune_prepare_kernel(
  m: Model,
  d: Data,
):
  worldId, i = wp.tid()  # Get the thread ID

  # Get the index of the current bounding box
  idx1 = d.box_sorting_indexer[worldId, i]

  end = d.box_projections_upper[worldId, idx1]
  limit = find_first_greater_than(worldId, d.box_projections_lower, end, i + 1, m.ngeom)
  limit = wp.min(m.ngeom - 1, limit)

  # Calculate the range of boxes for the sweep and prune process
  count = limit - i

  # Store the cumulative sum for the current box
  d.ranges[worldId, i] = count


@wp.func
def find_right_most_index_int(
  starts: wp.array(dtype=wp.int32, ndim=1), value: wp.int32, low: int, high: int
) -> int:
  while low < high:
    mid = (low + high) >> 1
    if starts[mid] > value:
      high = mid
    else:
      low = mid + 1
  return high


@wp.func
def find_indices(
  id: int, cumulative_sum: wp.array(dtype=wp.int32, ndim=1), length: int
) -> wp.vec2i:
  # Perform binary search to find the right most index
  i = find_right_most_index_int(cumulative_sum, id, 0, length)

  # Get the baseId, and compute the offset and j
  if i > 0:
    base_id = cumulative_sum[i - 1]
  else:
    base_id = 0
  offset = id - base_id
  j = i + offset + 1

  return wp.vec2i(i, j)


@wp.kernel
def broadphase_sweep_and_prune_kernel(
  m: Model, d: Data, num_threads: int, filter_parent: bool
):
  threadId = wp.tid()  # Get thread ID
  if d.cumulative_sum.shape[0] > 0:
    total_num_work_packages = d.cumulative_sum[d.cumulative_sum.shape[0] - 1]
  else:
    total_num_work_packages = 0

  while threadId < total_num_work_packages:
    # Get indices for current and next box pair
    ij = find_indices(threadId, d.cumulative_sum, d.cumulative_sum.shape[0])
    i = ij.x
    j = ij.y

    worldId = i // m.ngeom
    i = i % m.ngeom
    j = j % m.ngeom

    # geom index
    idx1 = d.box_sorting_indexer[worldId, i]
    idx2 = d.box_sorting_indexer[worldId, j]

    if not _geom_filter(m, idx1, idx2, filter_parent):
      threadId += num_threads
      continue

    """
    # somehow does not work yet.
    # exclude
    signature = (body1 << 16) + body2
    filtered = bool(False)
    # TODO(AD): this can become very expensive
    for i in range(m.nexclude):
      if m.exclude_signature[i] == signature:
        filtered = True
        break

    if filtered:
      threadId += num_threads
      continue
    """
    # Check if the boxes overlap
    if overlap(worldId, i, j, d.boxes_sorted):
      id = wp.atomic_add(d.broadphase_result_count, worldId, 1)

      if id < d.max_num_overlaps_per_world:
        d.broadphase_pairs[worldId, id] = _geom_pair(m, idx1, idx2)

    threadId += num_threads


@wp.kernel
def get_contact_solver_params_kernel(
  m: Model,
  d: Data,
):
  tid = wp.tid()

  n_contact_pts = d.ncon[0]
  if tid >= n_contact_pts:
    return

  geoms = d.contact.geom[tid]
  g1 = geoms.x
  g2 = geoms.y

  margin = wp.max(m.geom_margin[g1], m.geom_margin[g2])
  gap = wp.max(m.geom_gap[g1], m.geom_gap[g2])
  solmix1 = m.geom_solmix[g1]
  solmix2 = m.geom_solmix[g2]
  mix = solmix1 / (solmix1 + solmix2)
  mix = where((solmix1 < MJ_MINVAL) and (solmix2 < MJ_MINVAL), 0.5, mix)
  mix = where((solmix1 < MJ_MINVAL) and (solmix2 >= MJ_MINVAL), 0.0, mix)
  mix = where((solmix1 >= MJ_MINVAL) and (solmix2 < MJ_MINVAL), 1.0, mix)

  p1 = m.geom_priority[g1]
  p2 = m.geom_priority[g2]
  mix = where(p1 == p2, mix, where(p1 > p2, 1.0, 0.0))

  if m.geom_solref[g1].x > 0.0 and m.geom_solref[g2].x > 0.0:
    d.contact.solref[tid] = mix * m.geom_solref[g1] + (1.0 - mix) * m.geom_solref[g2]
  else:
    d.contact.solref[tid] = wp.min(m.geom_solref[g1], m.geom_solref[g2])
  d.contact.includemargin[tid] = margin - gap
  friction_ = wp.max(m.geom_friction[g1], m.geom_friction[g2])
  friction5 = vec5(friction_[0], friction_[0], friction_[1], friction_[2], friction_[2])
  d.contact.friction[tid] = friction5
  d.contact.solimp[tid] = mix * m.geom_solimp[g1] + (1.0 - mix) * m.geom_solimp[g2]


@wp.kernel
def group_contacts_by_type_kernel(
  m: Model,
  d: Data,
):
  worldid, tid = wp.tid()

  if tid >= d.broadphase_result_count[worldid]:
    return

  geoms = d.broadphase_pairs[worldid, tid]
  geom1 = geoms[0]
  geom2 = geoms[1]

  type1 = m.geom_type[geom1]
  type2 = m.geom_type[geom2]
  key = group_key(type1, type2)

  n_type_pair = wp.atomic_add(d.narrowphase_candidate_group_count, key, 1)
  d.narrowphase_candidate_worldid[key, n_type_pair] = worldid
  d.narrowphase_candidate_geom[key, n_type_pair] = wp.vec2i(geom1, geom2)


def broadphase_sweep_and_prune(m: Model, d: Data):
  """Broad-phase collision detection via sweep-and-prune."""

  # generate geom AABBs
  wp.launch(
    kernel=get_dyn_geom_aabb,
    dim=(d.nworld, m.ngeom),
    inputs=[m, d],
  )

  wp.launch(
    kernel=broadphase_project_boxes_onto_sweep_direction_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[d],
  )

  segmented_sort_available = hasattr(wp.utils, "segmented_sort_pairs")
  if segmented_sort_available:
    wp.utils.segmented_sort_pairs(
      d.box_projections_lower,
      d.box_sorting_indexer,
      m.ngeom * d.nworld,
      d.segment_indices,
    )
  else:
    # Sort each world's segment separately
    for world_id in range(d.nworld):
      start_idx = world_id * m.ngeom

      # Create temporary arrays for sorting
      temp_box_projections_lower = wp.zeros(
        m.ngeom * 2,
        dtype=d.box_projections_lower.dtype,
      )
      temp_box_sorting_indexer = wp.zeros(
        m.ngeom * 2,
        dtype=d.box_sorting_indexer.dtype,
      )

      # Copy data to temporary arrays
      wp.copy(
        temp_box_projections_lower,
        d.box_projections_lower,
        0,
        start_idx,
        m.ngeom,
      )
      wp.copy(
        temp_box_sorting_indexer,
        d.box_sorting_indexer,
        0,
        start_idx,
        m.ngeom,
      )

      # Sort the temporary arrays
      wp.utils.radix_sort_pairs(
        temp_box_projections_lower, temp_box_sorting_indexer, m.ngeom
      )

      # Copy sorted data back
      wp.copy(
        d.box_projections_lower,
        temp_box_projections_lower,
        start_idx,
        0,
        m.ngeom,
      )
      wp.copy(
        d.box_sorting_indexer,
        temp_box_sorting_indexer,
        start_idx,
        0,
        m.ngeom,
      )

  wp.launch(
    kernel=reorder_bounding_boxes_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[d],
  )

  wp.launch(
    kernel=broadphase_sweep_and_prune_prepare_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[m, d],
  )

  # The scan (scan = cumulative sum, either inclusive or exclusive depending on the last argument) is used for load balancing among the threads
  wp.utils.array_scan(d.ranges.reshape(-1), d.cumulative_sum, True)

  # Estimate how many overlap checks need to be done - assumes each box has to be compared to 5 other boxes (and batched over all worlds)
  num_sweep_threads = 5 * d.nworld * m.ngeom
  filter_parent = not m.opt.disableflags & DisableBit.FILTERPARENT.value
  wp.launch(
    kernel=broadphase_sweep_and_prune_kernel,
    dim=num_sweep_threads,
    inputs=[m, d, num_sweep_threads, filter_parent],
  )

  return d


def nxn_broadphase(m: Model, d: Data):
  filterparent = not (m.opt.disableflags & DisableBit.FILTERPARENT.value)

  d.broadphase_result_count.zero_()

  @wp.kernel
  def _nxn_broadphase(m: Model, d: Data):
    worldid, elementid = wp.tid()
    geom1 = (
      m.ngeom
      - 2
      - int(
        wp.sqrt(float(-8 * elementid + 4 * m.ngeom * (m.ngeom - 1) - 7)) / 2.0 - 0.5
      )
    )
    geom2 = (
      elementid
      + geom1
      + 1
      - m.ngeom * (m.ngeom - 1) // 2
      + (m.ngeom - geom1) * ((m.ngeom - geom1) - 1) // 2
    )

    margin1 = m.geom_margin[geom1]
    margin2 = m.geom_margin[geom2]
    pos1 = d.geom_xpos[worldid, geom1]
    pos2 = d.geom_xpos[worldid, geom2]
    xmat1 = d.geom_xmat[worldid, geom1]
    xmat2 = d.geom_xmat[worldid, geom2]
    size1 = m.geom_rbound[geom1]
    size2 = m.geom_rbound[geom2]

    bound = size1 + size2 + wp.max(margin1, margin2)
    dif = pos2 - pos1

    if size1 != 0.0 and size2 != 0.0:
      # neither geom is a plane
      dist_sq = wp.dot(dif, dif)
    elif size1 == 0.0:
      # geom1 is a plane
      dist_sq = wp.dot(dif, wp.vec3(xmat1[0, 2], xmat1[1, 2], xmat1[2, 2]))
    else:
      # geom2 is a plane
      dist_sq = wp.dot(-dif, wp.vec3(xmat2[0, 2], xmat2[1, 2], xmat2[2, 2]))

    bounds_filter = dist_sq <= bound * bound
    geom_filter = _geom_filter(m, geom1, geom2, filterparent)

    if bounds_filter and geom_filter:
      pairid = wp.atomic_add(d.broadphase_result_count, 0, 1)
      d.broadphase_pairs[worldid, pairid] = _geom_pair(m, geom1, geom2)

  wp.launch(
    _nxn_broadphase, dim=(d.nworld, m.ngeom * (m.ngeom - 1) // 2), inputs=[m, d]
  )


###########################################################################################3


def broadphase(m: Model, d: Data):
  # broadphase collision

  # TODO(team): determine ngeom to switch from n^2 to sap
  if m.ngeom <= 100:
    nxn_broadphase(m, d)
  else:
    broadphase_sweep_and_prune(m, d)


def group_contacts_by_type(m: Model, d: Data):
  # initialize type pair count & group contacts by type

  # Initialize type pair count
  d.narrowphase_candidate_group_count.zero_()

  wp.launch(
    group_contacts_by_type_kernel,
    dim=(d.nworld, d.max_num_overlaps_per_world),
    inputs=[m, d],
  )

  # Initialize the env contact counter
  d.ncon.zero_()


def get_contact_solver_params(m: Model, d: Data):
  wp.launch(
    get_contact_solver_params_kernel,
    dim=[d.nconmax],
    inputs=[m, d],
  )

  # TODO(team): do we need condim sorting, deepest penetrating contact here?


@event_scope
def collision(m: Model, d: Data):
  """Collision detection."""

  # AD: based on engine_collision_driver.py in Eric's warp fork/mjx-collisions-dev
  # which is further based on the CUDA code here:
  # https://github.com/btaba/mujoco/blob/warp-collisions/mjx/mujoco/mjx/_src/cuda/engine_collision_driver.cu.cc#L458-L583

  broadphase(m, d)
  # filtering?
  group_contacts_by_type(m, d)
  # XXX switch between collision functions and GJK/EPA here
  if True:
    from .collision_functions import narrowphase
  else:
    from .collision_convex import narrowphase

  # TODO(team): should we limit per-world contact nubmers?
  # TODO(team): we should reject far-away contacts in the narrowphase instead of constraint
  #             partitioning because we can move some pressure of the atomics
  narrowphase(m, d)
  get_contact_solver_params(m, d)
