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
from .types import Contact

BoxType = wp.types.matrix(shape=(2, 3), dtype=wp.float32)


@wp.func
def where(condition: bool, onTrue: float, onFalse: float) -> float:
    if condition:
        return onTrue
    else:
        return onFalse

@wp.func
def orthogonals(a: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
  """Returns orthogonal vectors `b` and `c`, given a vector `a`."""
  y = wp.vec3(0.0, 1.0, 0.0)
  z = wp.vec3(0.0, 0.0, 1.0)
  b = where(-0.5 < a[1] and a[1] < 0.5, y, z)
  b = b - a * wp.dot(a, b)
  # normalize b. however if a is a zero vector, zero b as well.
  b = wp.normalize(b) * float(wp.length(a) > 0.0)
  return b, wp.cross(a, b)


@wp.func
def make_frame(a: wp.vec3) -> wp.mat33:
  """Makes a right-handed 3D frame given a direction."""
  a = wp.normalize(a)
  b, c = orthogonals(a)
  return wp.mat33(a, b, c)




@wp.func
def _manifold_points(
  worldId : int,
  poly: wp.array(dtype=wp.vec3, ndim=2),
  poly_start: int,
  poly_count: int,
  poly_norm: wp.vec3,
  plane_pos: wp.vec3,
  n: wp.vec3,
  max_support: float,
) -> wp.vec4i:
  """Chooses four points on the polygon with approximately maximal area."""
  max_val = float(-1e6)
  a_idx = int(0)
  for i in range(poly_count):
    support = wp.dot(plane_pos - poly[worldId, poly_start+i], n)
    val = where(support > wp.max(0.0, max_support - 1e-3), 0.0, -1e6)
    if val > max_val:
      max_val = val
      a_idx = i
  a = poly[worldId, poly_start + a_idx]
  # choose point b furthest from a
  max_val = float(-1e6)
  b_idx = int(0)
  for i in range(poly_count):
    support = wp.dot(plane_pos - poly[worldId, poly_start+i], n)
    val = wp.length_sq(a - poly[worldId, poly_start + i]) + where(support > wp.max(0.0, max_support - 1e-3), 0.0, -1e6)
    if val > max_val:
      max_val = val
      b_idx = i
  b = poly[worldId, poly_start + b_idx]
  # choose point c furthest along the axis orthogonal to (a-b)
  ab = wp.cross(poly_norm, a - b)
  # ap = a - poly
  max_val = float(-1e6)
  c_idx = int(0)
  for i in range(poly_count):
    support = wp.dot(plane_pos - poly[worldId, poly_start+i], n)
    val = wp.abs(wp.dot(a - poly[worldId, poly_start + i], ab)) + where(support > wp.max(0.0, max_support - 1e-3), 0.0, -1e6)
    if val > max_val:
      max_val = val
      c_idx = i
  c = poly[worldId, poly_start + c_idx]
  # choose point d furthest from the other two triangle edges
  ac = wp.cross(poly_norm, a - c)
  bc = wp.cross(poly_norm, b - c)
  # bp = b - poly
  max_val = float(-1e6)
  d_idx = int(0)
  for i in range(poly_count):
    support = wp.dot(plane_pos - poly[worldId, poly_start+i], n)
    val = (
      wp.abs(wp.dot(b - poly[worldId, poly_start + i], bc))
      + wp.abs(wp.dot(a - poly[worldId, poly_start + i], ac))
      + where(support > wp.max(0.0, max_support - 1e-3), 0.0, -1e6)
    )
    if val > max_val:
      max_val = val
      d_idx = i
  return wp.vec4i(a_idx, b_idx, c_idx, d_idx)


@wp.func
def plane_convex(
  m: Model,
  d: Data,
  worldId : int,
  planeIndex: int,
  convexIndex: int,
  outBaseIndex: int,
  result: Contact,
):
  """Calculates contacts between a plane and a convex object."""
  vert_start = m.geom_vert_addr[worldId, convexIndex]
  vert_count = m.geom_vert_num[worldId, convexIndex]

  convexPos = d.geom_pos[worldId, convexIndex]
  convexMat = d.geom_mat[worldId, convexIndex]

  planePos = d.geom_pos[worldId, planeIndex]
  planeMat = d.geom_mat[worldId, planeIndex]

  # get points in the convex frame
  plane_pos = wp.transpose(convexMat) @ (planePos - convexPos)
  n = (
    wp.transpose(convexMat) @ planeMat[2]
  )  # TODO: Does [2] indeed return the last column of the matrix?

  max_support = float(-100000)
  for i in range(vert_count):
    max_support = wp.max(max_support, wp.dot(plane_pos - m.vert[worldId, vert_start+i], n))

  # search for manifold points within a 1mm skin depth
  idx = wp.vec4i(0)
  idx = _manifold_points(worldId, m.vert, vert_start, vert_count, n, plane_pos, n, max_support)
  frame = make_frame(
    wp.vec3(
      planeMat[0, 2], planeMat[1, 2], planeMat[2, 2]
    )
  )


  for i in range(4):
    # Get vertex position and convert to world frame
    id = int(idx[i])
    pos_i = m.vert[worldId, id]
    pos_i = convexPos + pos_i @ wp.transpose(convexMat)

    # Compute uniqueness by comparing with previous indices
    count = int(0)
    for j in range(i + 1):
      if idx[i] == idx[j]:
        count += 1
    unique = where(count == 1, 1.0, 0.0)

    # Compute distance and final position
    dist_i = where(unique > 0.0, -wp.dot(plane_pos - m.vert[worldId, vert_start+i], n), 1.0)
    pos_i = pos_i - 0.5 * dist_i * frame[2]

    # Store results
    result.dist[outBaseIndex + i] = dist_i
    result.pos[outBaseIndex + i] = pos_i
    result.frame[outBaseIndex + i] = frame

  # return ret




@wp.func
def pack_key(condim: int, g_min: int, g_max: int, local_id: int) -> int:
    return (condim << 28) | (g_min << 16) | (g_max << 3) | local_id

@wp.func
def decompose_key(key: int) -> wp.vec4i:
    # Extract components from key using bit masks and shifts
    condim = (key >> 28) & 0xF # 4 bits for condim
    g_min = (key >> 16) & 0xFFF # 12 bits for g_min
    g_max = (key >> 3) & 0x1FFF # 13 bits for g_max
    local_id = key & 7
    return wp.vec3i(condim, g_min, g_max, local_id)

