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

"""Tests for broad phase functions."""

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
from mujoco import mjx
import numpy as np
import warp as wp

from . import test_util

BoxType = wp.types.matrix(shape=(2, 3), dtype=wp.float32)


# Helper function to initialize a box
def init_box(min_x, min_y, min_z, max_x, max_y, max_z):
  center = wp.vec3((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
  size = wp.vec3(max_x - min_x, max_y - min_y, max_z - min_z)
  box = wp.types.matrix(shape=(2, 3), dtype=wp.float32)(
    [center.x, center.y, center.z, size.x, size.y, size.z]
  )
  return box


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


def transform_aabb(
  aabb: wp.types.matrix(shape=(2, 3), dtype=wp.float32),
  pos: wp.vec3,
  rot: wp.mat33,
) -> wp.types.matrix(shape=(2, 3), dtype=wp.float32):
  # Extract center and half-extents from AABB
  center = aabb[0]
  half_extents = aabb[1] * 0.5

  # Get absolute values of rotation matrix columns
  right = wp.vec3(wp.abs(rot[0, 0]), wp.abs(rot[0, 1]), wp.abs(rot[0, 2]))
  up = wp.vec3(wp.abs(rot[1, 0]), wp.abs(rot[1, 1]), wp.abs(rot[1, 2]))
  forward = wp.vec3(wp.abs(rot[2, 0]), wp.abs(rot[2, 1]), wp.abs(rot[2, 2]))

  # Compute world space half-extents
  world_extents = (
    right * half_extents.x + up * half_extents.y + forward * half_extents.z
  )

  # Transform center
  new_center = rot @ center + pos

  # Return new AABB as matrix with center and full size
  result = BoxType()
  result[0] = wp.vec3(new_center.x, new_center.y, new_center.z)
  result[1] = wp.vec3(
    world_extents.x * 2.0, world_extents.y * 2.0, world_extents.z * 2.0
  )
  return result


def find_overlaps_brute_force(worldId: int, num_boxes_per_world: int, boxes, pos, rot):
  """
  Finds overlapping bounding boxes using the brute-force O(n^2) algorithm.

  Returns:
      List of tuples [(idx1, idx2)] where idx1 and idx2 are indices of overlapping boxes.
  """
  overlaps = []

  for i in range(num_boxes_per_world):
    box_a = boxes[i]
    box_a = transform_aabb(box_a, pos[worldId, i], rot[worldId, i])

    for j in range(i + 1, num_boxes_per_world):
      box_b = boxes[j]
      box_b = transform_aabb(box_b, pos[worldId, j], rot[worldId, j])

      # Use the overlap function to check for overlap
      if overlap(box_a, box_b):
        overlaps.append((i, j))  # Store indices of overlapping boxes

  return overlaps


def find_overlaps_brute_force_batched(
  num_worlds: int, num_boxes_per_world: int, boxes, pos, rot
):
  """
  Finds overlapping bounding boxes using the brute-force O(n^2) algorithm.

  Returns:
      List of tuples [(idx1, idx2)] where idx1 and idx2 are indices of overlapping boxes.
  """

  overlaps = []

  for worldId in range(num_worlds):
    overlaps.append(
      find_overlaps_brute_force(worldId, num_boxes_per_world, boxes, pos, rot)
    )

  # Show progress bar for brute force computation
  # from tqdm import tqdm

  # for worldId in tqdm(range(num_worlds), desc="Computing overlaps"):
  #    overlaps.append(find_overlaps_brute_force(worldId, num_boxes_per_world, boxes))

  return overlaps


class MultiIndexList:
  def __init__(self):
    self.data = {}

  def __setitem__(self, key, value):
    worldId, i = key
    if worldId not in self.data:
      self.data[worldId] = []
    if i >= len(self.data[worldId]):
      self.data[worldId].extend([None] * (i - len(self.data[worldId]) + 1))
    self.data[worldId][i] = value

  def __getitem__(self, key):
    worldId, i = key
    return self.data[worldId][i]  # Raises KeyError if not found


class BroadPhaseTest(parameterized.TestCase):
  def test_broad_phase(self):
    """Tests broad phase."""
    _, mjd, m, d = test_util.fixture("humanoid/humanoid.xml")

    # Create some test boxes
    num_worlds = d.nworld
    num_boxes_per_world = m.ngeom
    # print(f"num_worlds: {num_worlds}, num_boxes_per_world: {num_boxes_per_world}")

    # Parameters for random box generation
    sample_space_origin = wp.vec3(-10.0, -10.0, -10.0)  # Origin of the bounding volume
    sample_space_size = wp.vec3(20.0, 20.0, 20.0)  # Size of the bounding volume
    min_edge_length = 0.5  # Minimum edge length of random boxes
    max_edge_length = 5.0  # Maximum edge length of random boxes

    boxes_list = []

    # Set random seed for reproducibility
    import random

    random.seed(11)

    # Generate random boxes for each world
    for _ in range(num_boxes_per_world):
      # Generate random position within bounding volume
      pos_x = sample_space_origin.x + random.random() * sample_space_size.x
      pos_y = sample_space_origin.y + random.random() * sample_space_size.y
      pos_z = sample_space_origin.z + random.random() * sample_space_size.z

      # Generate random box dimensions between min and max edge lengths
      size_x = min_edge_length + random.random() * (max_edge_length - min_edge_length)
      size_y = min_edge_length + random.random() * (max_edge_length - min_edge_length)
      size_z = min_edge_length + random.random() * (max_edge_length - min_edge_length)

      # Create box with random position and size
      boxes_list.append(
        init_box(pos_x, pos_y, pos_z, pos_x + size_x, pos_y + size_y, pos_z + size_z)
      )

    # Generate random positions and orientations for each box
    pos = []
    rot = []
    for _ in range(num_worlds * num_boxes_per_world):
      # Random position within bounding volume
      pos_x = sample_space_origin.x + random.random() * sample_space_size.x
      pos_y = sample_space_origin.y + random.random() * sample_space_size.y
      pos_z = sample_space_origin.z + random.random() * sample_space_size.z
      pos.append(wp.vec3(pos_x, pos_y, pos_z))
      # pos.append(wp.vec3(0, 0, 0))

      # Random rotation matrix
      rx = random.random() * 6.28318530718  # 2*pi
      ry = random.random() * 6.28318530718
      rz = random.random() * 6.28318530718
      axis = wp.vec3(rx, ry, rz)
      axis = axis / wp.length(axis)  # normalize axis
      angle = random.random() * 6.28318530718  # random angle between 0 and 2*pi
      rot.append(wp.quat_to_matrix(wp.quat_from_axis_angle(axis, angle)))
      # rot.append(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(1, 0, 0), float(0))))

    # Convert pos and rot to MultiIndexList format
    pos_multi = MultiIndexList()
    rot_multi = MultiIndexList()

    # Populate the MultiIndexLists using pos and rot data
    idx = 0
    for world_idx in range(num_worlds):
      for i in range(num_boxes_per_world):
        pos_multi[world_idx, i] = pos[idx]
        rot_multi[world_idx, i] = rot[idx]
        idx += 1

    brute_force_overlaps = find_overlaps_brute_force_batched(
      num_worlds, num_boxes_per_world, boxes_list, pos_multi, rot_multi
    )

    # Test the broad phase by setting custom aabb data
    d.geom_aabb = wp.array(
      boxes_list, dtype=wp.types.matrix(shape=(2, 3), dtype=wp.float32)
    )
    d.geom_aabb = d.geom_aabb.reshape((num_boxes_per_world))
    d.geom_xpos = wp.array(pos, dtype=wp.vec3)
    d.geom_xpos = d.geom_xpos.reshape((num_worlds, num_boxes_per_world))
    d.geom_xmat = wp.array(rot, dtype=wp.mat33)
    d.geom_xmat = d.geom_xmat.reshape((num_worlds, num_boxes_per_world))

    mjx.broad_phase(m, d)

    result = d.broadphase_pairs
    result_count = d.result_count

    # Get numpy arrays from result and result_count
    result_np = result.numpy()
    result_count_np = result_count.numpy()

    # Iterate over each world
    for world_idx in range(num_worlds):
      # Get number of collisions for this world
      num_collisions = result_count_np[world_idx]
      print(f"Number of collisions for world {world_idx}: {num_collisions}")

      list = brute_force_overlaps[world_idx]
      assert len(list) == num_collisions, "Number of collisions does not match"

      # Print each collision pair
      for i in range(num_collisions):
        pair = result_np[world_idx][i]

        # Convert pair to tuple for comparison
        pair_tuple = (int(pair[0]), int(pair[1]))
        assert pair_tuple in list, (
          f"Collision pair {pair_tuple} not found in brute force results"
        )


if __name__ == "__main__":
  wp.init()
  absltest.main()
