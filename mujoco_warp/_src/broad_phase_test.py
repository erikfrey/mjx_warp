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

"""Tests for broad phase functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjwarp

from . import collision_driver
from . import test_util


@wp.func
def xyz(v: wp.vec4) -> wp.vec3:
  return wp.vec3(v.x, v.y, v.z)


def overlap(
  worldId: int, i_a: int, i_b: int, geom_xpos, geom_xmat, geom_rbound, geom_margin
) -> bool:
  r_a = geom_rbound[i_a]
  r_b = geom_rbound[i_b]
  margin_a = geom_margin[i_a]
  margin_b = geom_margin[i_b]
  pos_a = geom_xpos[worldId][i_a]
  pos_b = geom_xpos[worldId][i_b]

  if r_a == 0.0 and r_b == 0.0:
    # both are planes
    return False
  elif r_a == 0.0 or r_b == 0.0:
    if r_b == 0.0:  # swap if required such that a is always a plane
      tmp_pos = pos_a
      tmp_r = r_a
      tmp_margin = margin_a
      pos_a = pos_b
      r_a = r_b
      margin_a = margin_b
      pos_b = tmp_pos
      r_b = tmp_r
      margin_b = tmp_margin
      plane_normal = wp.vec3(
        geom_xmat[worldId][i_a, 0, 2],
        geom_xmat[worldId][i_a, 1, 2],
        geom_xmat[worldId][i_a, 2, 2],
      )
    else:
      plane_normal = wp.vec3(
        geom_xmat[worldId][i_b, 0, 2],
        geom_xmat[worldId][i_b, 1, 2],
        geom_xmat[worldId][i_b, 2, 2],
      )

    # Check plane-sphere intersection
    delta = pos_b - pos_a
    dist = wp.dot(delta, plane_normal)
    return dist <= (r_b + margin_a + margin_b)
  else:
    # both are spheres
    delta = pos_a - pos_b
    dist_sq = wp.dot(delta, delta)
    radius_sum = r_a + r_b + margin_a + margin_b
    return dist_sq <= radius_sum * radius_sum


def find_overlaps_brute_force(
  worldId: int, num_boxes_per_world: int, xpos, xmat, radius, margin, geom_bodyid
):
  """
  Finds overlapping bounding boxes using the brute-force O(n^2) algorithm.
  Returns:
      List of tuples [(idx1, idx2)] where idx1 and idx2 are indices of overlapping boxes.
  """
  overlaps = []

  for i in range(num_boxes_per_world):
    for j in range(i + 1, num_boxes_per_world):
      if geom_bodyid[i] == geom_bodyid[j]:
        continue

      if overlap(worldId, i, j, xpos, xmat, radius, margin):
        overlaps.append((i, j))  # Store indices of overlapping boxes

  return overlaps


def find_overlaps_brute_force_batched(
  num_worlds: int, num_boxes_per_world: int, xpos, xmat, radius, margin, geom_bodyid
):
  """
  Finds overlapping bounding boxes using the brute-force O(n^2) algorithm.
  Returns:
      List of tuples [(idx1, idx2)] where idx1 and idx2 are indices of overlapping boxes.
  """

  overlaps = []

  for worldId in range(num_worlds):
    overlaps.append(
      find_overlaps_brute_force(
        worldId, num_boxes_per_world, xpos, xmat, radius, margin, geom_bodyid
      )
    )

  return overlaps


class BroadPhaseTest(parameterized.TestCase):
  def test_broadphase_sweep_and_prune(self):
    """Tests broadphase_sweep_and_prune."""

    _MODEL = """
     <mujoco>
      <worldbody>
        <geom size="40 40 40" type="plane"/>   <!- (0) intersects with nothing -->
        <body pos="0 0 0.7">
          <freejoint/>
          <geom size="0.5 0.5 0.5" type="box"/> <!- (1) intersects with 2, 6, 7 -->
        </body>
        <body pos="0.1 0 0.7">
          <freejoint/>
          <geom size="0.5 0.5 0.5" type="box"/> <!- (2) intersects with 1, 6, 7 -->
        </body>

        <body pos="1.8 0 0.7">
          <freejoint/>
          <geom size="0.5 0.5 0.5" type="box"/> <!- (3) intersects with 4  -->
        </body>
        <body pos="1.6 0 0.7">
          <freejoint/>
          <geom size="0.5 0.5 0.5" type="box"/> <!- (4) intersects with 3 -->
        </body>

        <body pos="0 0 1.8">
          <freejoint/>
          <geom size="0.5 0.5 0.5" type="box"/> <!- (5) intersects with 7 -->
          <geom size="0.5 0.5 0.5" type="box" pos="0 0 -1"/> <!- (6) intersects with 2, 1, 7 -->
        </body>
        <body pos="0 0.5 1.2">
          <freejoint/>
          <geom size="0.5 0.5 0.5" type="box"/> <!- (7) intersects with 5, 6 -->
        </body>
        
      </worldbody>
    </mujoco>
    """

    m = mujoco.MjModel.from_xml_string(_MODEL)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    mx = mjwarp.put_model(m)
    dx = mjwarp.put_data(m, d)

    mjwarp.broadphase_sweep_and_prune(mx, dx)

    m = mx
    d = dx
    pos = d.geom_xpos.numpy()
    mat = d.geom_xmat.numpy()

    pos = pos.reshape((d.nworld, m.ngeom, 3))
    mat = mat.reshape((d.nworld, m.ngeom, 3, 3))

    brute_force_overlaps = find_overlaps_brute_force_batched(
      d.nworld,
      m.ngeom,
      pos,
      mat,
      m.geom_rbound.numpy(),
      m.geom_margin.numpy(),
      m.geom_bodyid.numpy(),
    )

    ncollision = dx.ncollision.numpy()[0]
    np.testing.assert_equal(ncollision, len(brute_force_overlaps[0]), "ncollision")

    mjwarp.broadphase_sweep_and_prune(m, d)

    # Get numpy arrays from result and ncollision
    result_np = d.collision_pair.numpy()
    worldid_np = d.collision_worldid.numpy()

    # Iterate over each world
    for world_idx in range(d.nworld):
      # Get number of collisions for this world
      num_collisions = np.sum(worldid_np[:ncollision] == world_idx)
      print(f"num_collisions: {num_collisions}")
      list = brute_force_overlaps[world_idx]
      np.testing.assert_equal(len(list), num_collisions, "num_collisions")

      # Print each collision pair
      for i in range(ncollision):
        if worldid_np[i] != world_idx:
          continue
        pair = result_np[i]

        # Convert pair to tuple for comparison
        # TODO(team): confirm ordering is correct
        if pair[0] > pair[1]:
          pair_tuple = (int(pair[1]), int(pair[0]))
        else:
          pair_tuple = (int(pair[0]), int(pair[1]))
        np.testing.assert_equal(
          pair_tuple in list,
          True,
          f"Collision pair {pair_tuple} not found in brute force results",
        )

  def test_nxn_broadphase(self):
    """Tests nxn_broadphase."""
    # one world and zero collisions
    mjm, _, m, d0 = test_util.fixture("broadphase.xml", keyframe=0)
    collision_driver.nxn_broadphase(m, d0)
    np.testing.assert_allclose(d0.ncollision.numpy()[0], 0)

    # one world and one collision
    _, mjd1, _, d1 = test_util.fixture("broadphase.xml", keyframe=1)
    collision_driver.nxn_broadphase(m, d1)

    np.testing.assert_allclose(d1.ncollision.numpy()[0], 1)
    np.testing.assert_allclose(d1.collision_pair.numpy()[0][0], 0)
    np.testing.assert_allclose(d1.collision_pair.numpy()[0][1], 1)

    # one world and three collisions
    _, mjd2, _, d2 = test_util.fixture("broadphase.xml", keyframe=2)
    collision_driver.nxn_broadphase(m, d2)
    np.testing.assert_allclose(d2.ncollision.numpy()[0], 3)
    np.testing.assert_allclose(d2.collision_pair.numpy()[0][0], 0)
    np.testing.assert_allclose(d2.collision_pair.numpy()[0][1], 1)
    np.testing.assert_allclose(d2.collision_pair.numpy()[1][0], 0)
    np.testing.assert_allclose(d2.collision_pair.numpy()[1][1], 2)
    np.testing.assert_allclose(d2.collision_pair.numpy()[2][0], 1)
    np.testing.assert_allclose(d2.collision_pair.numpy()[2][1], 2)

    # two worlds and four collisions
    d3 = mjwarp.make_data(mjm, nworld=2)
    d3.geom_xpos = wp.array(
      np.vstack(
        [np.expand_dims(mjd1.geom_xpos, axis=0), np.expand_dims(mjd2.geom_xpos, axis=0)]
      ),
      dtype=wp.vec3,
    )

    collision_driver.nxn_broadphase(m, d3)
    np.testing.assert_allclose(d3.ncollision.numpy()[0], 4)
    np.testing.assert_allclose(d3.collision_pair.numpy()[0][0], 0)
    np.testing.assert_allclose(d3.collision_pair.numpy()[0][1], 1)
    np.testing.assert_allclose(d3.collision_pair.numpy()[1][0], 0)
    np.testing.assert_allclose(d3.collision_pair.numpy()[1][1], 1)
    np.testing.assert_allclose(d3.collision_pair.numpy()[2][0], 0)
    np.testing.assert_allclose(d3.collision_pair.numpy()[2][1], 2)
    np.testing.assert_allclose(d3.collision_pair.numpy()[3][0], 1)
    np.testing.assert_allclose(d3.collision_pair.numpy()[3][1], 2)

    # one world and zero collisions: contype and conaffinity incompatibility
    _, _, m4, d4 = test_util.fixture("broadphase.xml", keyframe=1)
    m4.geom_contype = wp.array(np.array([0, 0, 0]), dtype=wp.int32)
    m4.geom_conaffinity = wp.array(np.array([1, 1, 1]), dtype=wp.int32)
    collision_driver.nxn_broadphase(m4, d4)
    np.testing.assert_allclose(d4.ncollision.numpy()[0], 0)

    # one world and one collision: geomtype ordering
    _, _, _, d5 = test_util.fixture("broadphase.xml", keyframe=3)
    collision_driver.nxn_broadphase(m, d5)
    np.testing.assert_allclose(d5.ncollision.numpy()[0], 1)
    np.testing.assert_allclose(d5.collision_pair.numpy()[0][0], 3)
    np.testing.assert_allclose(d5.collision_pair.numpy()[0][1], 2)

    # TODO(team): test margin


if __name__ == "__main__":
  wp.init()
  absltest.main()
