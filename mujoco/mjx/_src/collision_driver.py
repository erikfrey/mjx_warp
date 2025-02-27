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

from .types import Model
from .types import Data

BoxType = wp.types.matrix(shape=(2, 3), dtype=wp.float32)


# TODO: Verify that this is corect
@wp.func
def transform_aabb(
  aabb: wp.types.matrix(shape=(2, 3), dtype=wp.float32),
  pos: wp.vec3,
  rot: wp.mat33,
) -> wp.types.matrix(shape=(2, 3), dtype=wp.float32):
  # Extract center and extents from AABB
  center = aabb[0]
  extents = aabb[1]

  absRot = rot
  absRot[0, 0] = wp.abs(rot[0, 0])
  absRot[0, 1] = wp.abs(rot[0, 1])
  absRot[0, 2] = wp.abs(rot[0, 2])
  absRot[1, 0] = wp.abs(rot[1, 0])
  absRot[1, 1] = wp.abs(rot[1, 1])
  absRot[1, 2] = wp.abs(rot[1, 2])
  absRot[2, 0] = wp.abs(rot[2, 0])
  absRot[2, 1] = wp.abs(rot[2, 1])
  absRot[2, 2] = wp.abs(rot[2, 2])
  world_extents = extents * absRot

  # Transform center
  new_center = rot @ center + pos

  # Return new AABB as matrix with center and full size
  result = BoxType()
  result[0] = wp.vec3(new_center.x, new_center.y, new_center.z)
  result[1] = wp.vec3(world_extents.x, world_extents.y, world_extents.z)
  return result


@wp.func
def overlap(
  a: wp.types.matrix(shape=(2, 3), dtype=wp.float32),
  b: wp.types.matrix(shape=(2, 3), dtype=wp.float32),
) -> bool:
  # Extract centers and sizes
  a_center = a[0]
  a_size = a[1]
  b_center = b[0]
  b_size = b[1]

  # Calculate min/max from center and size
  a_min = a_center - 0.5 * a_size
  a_max = a_center + 0.5 * a_size
  b_min = b_center - 0.5 * b_size
  b_max = b_center + 0.5 * b_size

  return not (
    a_min.x > b_max.x
    or b_min.x > a_max.x
    or a_min.y > b_max.y
    or b_min.y > a_max.y
    or a_min.z > b_max.z
    or b_min.z > a_max.z
  )


@wp.kernel
def broad_phase_project_boxes_onto_sweep_direction_kernel(
  boxes: wp.array(dtype=wp.types.matrix(shape=(2, 3), dtype=wp.float32), ndim=1),
  box_translations: wp.array(dtype=wp.vec3, ndim=2),
  box_rotations: wp.array(dtype=wp.mat33, ndim=2),
  data_start: wp.array(dtype=wp.float32, ndim=2),
  data_end: wp.array(dtype=wp.float32, ndim=2),
  data_indexer: wp.array(dtype=wp.int32, ndim=2),
  direction: wp.vec3,
  abs_dir: wp.vec3,
  result_count: wp.array(dtype=wp.int32, ndim=1),
):
  worldId, i = wp.tid()

  box = boxes[i]  # box is a vector6
  box = transform_aabb(box, box_translations[worldId, i], box_rotations[worldId, i])
  box_center = box[0]
  box_size = box[1]
  center = wp.dot(direction, box_center)
  d = wp.dot(box_size, abs_dir)
  f = center - d

  # Store results in the data arrays
  data_start[worldId, i] = f
  data_end[worldId, i] = center + d
  data_indexer[worldId, i] = i

  if i == 0:
    result_count[worldId] = 0  # Initialize result count to 0


@wp.kernel
def reorder_bounding_boxes_kernel(
  boxes: wp.array(dtype=wp.types.matrix(shape=(2, 3), dtype=wp.float32), ndim=1),
  box_translations: wp.array(dtype=wp.vec3, ndim=2),
  box_rotations: wp.array(dtype=wp.mat33, ndim=2),
  boxes_sorted: wp.array(dtype=wp.types.matrix(shape=(2, 3), dtype=wp.float32), ndim=2),
  data_indexer: wp.array(dtype=wp.int32, ndim=2),
):
  worldId, i = wp.tid()

  # Get the index from the data indexer
  mapped = data_indexer[worldId, i]

  # Get the box from the original boxes array
  box = boxes[mapped]
  box = transform_aabb(
    box, box_translations[worldId, mapped], box_rotations[worldId, mapped]
  )

  # Reorder the box into the sorted array
  boxes_sorted[worldId, i] = box


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
def broad_phase_sweep_and_prune_prepare_kernel(
  num_boxes_per_world: int,
  data_start: wp.array(dtype=wp.float32, ndim=2),
  data_end: wp.array(dtype=wp.float32, ndim=2),
  indexer: wp.array(dtype=wp.int32, ndim=2),
  cumulative_sum: wp.array(dtype=wp.int32, ndim=2),
):
  worldId, i = wp.tid()  # Get the thread ID

  # Get the index of the current bounding box
  idx1 = indexer[worldId, i]

  end = data_end[worldId, idx1]
  limit = find_first_greater_than(worldId, data_start, end, i + 1, num_boxes_per_world)
  limit = wp.min(num_boxes_per_world - 1, limit)

  # Calculate the range of boxes for the sweep and prune process
  count = limit - i

  # Store the cumulative sum for the current box
  cumulative_sum[worldId, i] = count


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
def broad_phase_sweep_and_prune_kernel(
  num_threads: int,
  length: int,
  num_boxes_per_world: int,
  max_num_overlaps_per_world: int,
  cumulative_sum: wp.array(dtype=wp.int32, ndim=1),
  data_indexer: wp.array(dtype=wp.int32, ndim=2),
  data_result: wp.array(dtype=wp.vec2i, ndim=2),
  result_count: wp.array(dtype=wp.int32, ndim=1),
  boxes_sorted: wp.array(dtype=wp.types.matrix(shape=(2, 3), dtype=wp.float32), ndim=2),
):
  threadId = wp.tid()  # Get thread ID
  if length > 0:
    total_num_work_packages = cumulative_sum[length - 1]
  else:
    total_num_work_packages = 0

  while threadId < total_num_work_packages:
    # Get indices for current and next box pair
    ij = find_indices(threadId, cumulative_sum, length)
    i = ij.x
    j = ij.y

    worldId = i // num_boxes_per_world
    i = i % num_boxes_per_world

    # world_id_j = j // num_boxes_per_world
    j = j % num_boxes_per_world

    # assert worldId == world_id_j, "Only boxes in the same world can be compared"
    # TODO: Remove print if debugging is done
    # if worldId != world_id_j:
    #     print("Only boxes in the same world can be compared")

    idx1 = data_indexer[worldId, i]

    box1 = boxes_sorted[worldId, i]

    idx2 = data_indexer[worldId, j]

    # Check if the boxes overlap
    if idx1 != idx2 and overlap(box1, boxes_sorted[worldId, j]):
      pair = wp.vec2i(wp.min(idx1, idx2), wp.max(idx1, idx2))

      id = wp.atomic_add(result_count, worldId, 1)

      if id < max_num_overlaps_per_world:
        data_result[worldId, id] = pair

    threadId += num_threads


def broad_phase(m: Model, d: Data) -> Data:
  """Broad phase collision detection."""

  # Directional vectors for sweep
  # TODO: Improve picking of direction
  direction = wp.vec3(0.5935, 0.7790, 0.1235)
  direction = wp.normalize(direction)
  abs_dir = wp.vec3(abs(direction.x), abs(direction.y), abs(direction.z))

  wp.launch(
    kernel=broad_phase_project_boxes_onto_sweep_direction_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[
      d.geom_aabb,
      d.geom_xpos,
      d.geom_xmat,
      d.data_start,
      d.data_end,
      d.data_indexer,
      direction,
      abs_dir,
      d.result_count,
    ],
  )

  segmented_sort_available = hasattr(wp.utils, "segmented_sort_pairs")

  if segmented_sort_available:
    # print("Using segmented sort")
    wp.utils.segmented_sort_pairs(
      d.data_start,
      d.data_indexer,
      m.ngeom * d.nworld,
      d.segment_indices,
      d.nworld,
    )
  else:
    # Sort each world's segment separately
    for world_id in range(d.nworld):
      start_idx = world_id * m.ngeom

      # Create temporary arrays for sorting
      temp_data_start = wp.zeros(
        m.ngeom * 2,
        dtype=d.data_start.dtype,
      )
      temp_data_indexer = wp.zeros(
        m.ngeom * 2,
        dtype=d.data_indexer.dtype,
      )

      # Copy data to temporary arrays
      wp.copy(
        temp_data_start,
        d.data_start,
        0,
        start_idx,
        m.ngeom,
      )
      wp.copy(
        temp_data_indexer,
        d.data_indexer,
        0,
        start_idx,
        m.ngeom,
      )

      # Sort the temporary arrays
      wp.utils.radix_sort_pairs(temp_data_start, temp_data_indexer, m.ngeom)

      # Copy sorted data back
      wp.copy(
        d.data_start,
        temp_data_start,
        start_idx,
        0,
        m.ngeom,
      )
      wp.copy(
        d.data_indexer,
        temp_data_indexer,
        start_idx,
        0,
        m.ngeom,
      )

  wp.launch(
    kernel=reorder_bounding_boxes_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[d.geom_aabb, d.geom_xpos, d.geom_xmat, d.boxes_sorted, d.data_indexer],
  )

  wp.launch(
    kernel=broad_phase_sweep_and_prune_prepare_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[
      m.ngeom,
      d.data_start,
      d.data_end,
      d.data_indexer,
      d.ranges,
    ],
  )

  # The scan (scan = cumulative sum, either inclusive or exclusive depending on the last argument) is used for load balancing among the threads
  wp.utils.array_scan(d.ranges.reshape(-1), d.cumulative_sum, True)

  # Estimate how many overlap checks need to be done - assumes each box has to be compared to 5 other boxes (and batched over all worlds)
  num_sweep_threads = 5 * d.nworld * m.ngeom
  wp.launch(
    kernel=broad_phase_sweep_and_prune_kernel,
    dim=num_sweep_threads,
    inputs=[
      num_sweep_threads,
      d.nworld * m.ngeom,
      m.ngeom,
      d.max_num_overlaps_per_world,
      d.cumulative_sum,
      d.data_indexer,
      d.broadphase_pairs,
      d.result_count,
      d.boxes_sorted,
    ],
  )

  return d
