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


import itertools
import math
from typing import Any

import numpy as np
import warp as wp

from .types import Model
from .types import Data
from .types import GeomType
from .math import make_frame
from .math import closest_segment_to_segment_points
from .math import normalize_with_norm
from .support import group_key
from .support import mat33_from_cols


@wp.struct
class GeomPlane:
  pos: wp.vec3
  rot: wp.mat33
  normal: wp.vec3


@wp.struct
class GeomSphere:
  pos: wp.vec3
  rot: wp.mat33
  radius: float


@wp.struct
class GeomCapsule:
  pos: wp.vec3
  rot: wp.mat33
  radius: float
  halfsize: float


@wp.struct
class GeomEllipsoid:
  pos: wp.vec3
  rot: wp.mat33
  size: wp.vec3


@wp.struct
class GeomCylinder:
  pos: wp.vec3
  rot: wp.mat33
  radius: float
  halfsize: float


@wp.struct
class GeomBox:
  pos: wp.vec3
  rot: wp.mat33
  size: wp.vec3


@wp.struct
class GeomMesh:
  pos: wp.vec3
  rot: wp.mat33
  vertadr: int
  vertnum: int


def get_info(t):
  @wp.func
  def _get_info(
    gid: int,
    m: Model,
    geom_xpos: wp.array(dtype=wp.vec3),
    geom_xmat: wp.array(dtype=wp.mat33),
  ):
    pos = geom_xpos[gid]
    rot = geom_xmat[gid]
    size = m.geom_size[gid]
    if wp.static(t == GeomType.SPHERE.value):
      sphere = GeomSphere()
      sphere.pos = pos
      sphere.rot = rot
      sphere.radius = size[0]
      return sphere
    elif wp.static(t == GeomType.BOX.value):
      box = GeomBox()
      box.pos = pos
      box.rot = rot
      box.size = size
      return box
    elif wp.static(t == GeomType.PLANE.value):
      plane = GeomPlane()
      plane.pos = pos
      plane.rot = rot
      plane.normal = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
      return plane
    elif wp.static(t == GeomType.CAPSULE.value):
      capsule = GeomCapsule()
      capsule.pos = pos
      capsule.rot = rot
      capsule.radius = size[0]
      capsule.halfsize = size[1]
      return capsule
    elif wp.static(t == GeomType.ELLIPSOID.value):
      ellipsoid = GeomEllipsoid()
      ellipsoid.pos = pos
      ellipsoid.rot = rot
      ellipsoid.size = size
      return ellipsoid
    elif wp.static(t == GeomType.CYLINDER.value):
      cylinder = GeomCylinder()
      cylinder.pos = pos
      cylinder.rot = rot
      cylinder.radius = size[0]
      cylinder.halfsize = size[1]
      return cylinder
    elif wp.static(t == GeomType.MESH.value):
      mesh = GeomMesh()
      mesh.pos = pos
      mesh.rot = rot
      dataid = m.geom_dataid[gid]
      if dataid >= 0:
        mesh.vertadr = m.mesh_vertadr[dataid]
        mesh.vertnum = m.mesh_vertnum[dataid]
      else:
        mesh.vertadr = 0
        mesh.vertnum = 0
      return mesh
    else:
      wp.static(RuntimeError("Unsupported type", t))

  return _get_info


@wp.func
def write_contact(
  d: Data,
  dist: float,
  pos: wp.vec3,
  frame: wp.mat33,
  margin: float,
  geoms: wp.vec2i,
  worldid: int,
):
  active = (dist - margin) < 0
  if active:
    index = wp.atomic_add(d.ncon, 0, 1)
    if index < d.nconmax:
      d.contact.dist[index] = dist
      d.contact.pos[index] = pos
      d.contact.frame[index] = frame
      d.contact.geom[index] = geoms
      d.contact.worldid[index] = worldid


@wp.func
def _plane_sphere(
  plane_normal: wp.vec3, plane_pos: wp.vec3, sphere_pos: wp.vec3, sphere_radius: float
):
  dist = wp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius
  pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
  return dist, pos


@wp.func
def plane_sphere(
  plane: GeomPlane,
  sphere: GeomSphere,
  worldid: int,
  d: Data,
  margin: float,
  geom_indices: wp.vec2i,
):
  dist, pos = _plane_sphere(plane.normal, plane.pos, sphere.pos, sphere.radius)

  write_contact(d, dist, pos, make_frame(plane.normal), margin, geom_indices, worldid)


@wp.func
def _sphere_sphere(
  pos1: wp.vec3,
  radius1: float,
  pos2: wp.vec3,
  radius2: float,
  worldid: int,
  d: Data,
  margin: float,
  geom_indices: wp.vec2i,
):
  dir = pos2 - pos1
  dist = wp.length(dir)
  if dist == 0.0:
    n = wp.vec3(1.0, 0.0, 0.0)
  else:
    n = dir / dist
  dist = dist - (radius1 + radius2)
  pos = pos1 + n * (radius1 + 0.5 * dist)

  write_contact(d, dist, pos, make_frame(n), margin, geom_indices, worldid)


@wp.func
def sphere_sphere(
  sphere1: GeomSphere,
  sphere2: GeomSphere,
  worldid: int,
  d: Data,
  margin: float,
  geom_indices: wp.vec2i,
):
  _sphere_sphere(
    sphere1.pos,
    sphere1.radius,
    sphere2.pos,
    sphere2.radius,
    worldid,
    d,
    margin,
    geom_indices,
  )


@wp.func
def capsule_capsule(
  cap1: GeomCapsule,
  cap2: GeomCapsule,
  worldid: int,
  d: Data,
  margin: float,
  geom_indices: wp.vec2i,
):
  axis1 = wp.vec3(cap1.rot[0, 2], cap1.rot[1, 2], cap1.rot[2, 2])
  axis2 = wp.vec3(cap2.rot[0, 2], cap2.rot[1, 2], cap2.rot[2, 2])
  length1 = cap1.halfsize
  length2 = cap2.halfsize
  seg1 = axis1 * length1
  seg2 = axis2 * length2

  pt1, pt2 = closest_segment_to_segment_points(
    cap1.pos - seg1,
    cap1.pos + seg1,
    cap2.pos - seg2,
    cap2.pos + seg2,
  )

  _sphere_sphere(pt1, cap1.radius, pt2, cap2.radius, worldid, d, margin, geom_indices)


@wp.func
def plane_capsule(
  plane: GeomPlane,
  cap: GeomCapsule,
  worldid: int,
  d: Data,
  margin: float,
  geom_indices: wp.vec2i,
):
  """Calculates two contacts between a capsule and a plane."""
  n = plane.normal
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  # align contact frames with capsule axis
  b, b_norm = normalize_with_norm(axis - n * wp.dot(n, axis))

  if b_norm < 0.5:
    if -0.5 < n[1] and n[1] < 0.5:
      b = wp.vec3(0.0, 1.0, 0.0)
    else:
      b = wp.vec3(0.0, 0.0, 1.0)

  frame = mat33_from_cols(n, b, wp.cross(n, b))
  segment = axis * cap.halfsize

  dist1, pos1 = _plane_sphere(n, plane.pos, cap.pos + segment, cap.radius)
  write_contact(d, dist1, pos1, frame, margin, geom_indices, worldid)

  dist2, pos2 = _plane_sphere(n, plane.pos, cap.pos - segment, cap.radius)
  write_contact(d, dist2, pos2, frame, margin, geom_indices, worldid)


HUGE_VAL = 1e6
TINY_VAL = 1e-6

# BOX_VERTS = np.array(list(itertools.product((-1, 1), (-1, 1), (-1, 1))), dtype=float)
#
# BOX_FACES = np.array([
#       0, 4, 5, 1,
#       0, 2, 6, 4,
#       6, 7, 5, 4,
#       2, 3, 7, 6,
#       1, 5, 7, 3,
#       0, 1, 3, 2,
#   ]).reshape((-1, 4))  # fmt: skip


class mat83f(wp.types.matrix(shape=(8, 3), dtype=wp.float32)):
  pass


class mat16_3f(wp.types.matrix(shape=(16, 3), dtype=wp.float32)):
  pass


class mat34f(wp.types.matrix(shape=(3, 4), dtype=wp.float32)):
  pass


class mat43f(wp.types.matrix(shape=(4, 3), dtype=wp.float32)):
  pass


class mat38f(wp.types.matrix(shape=(3, 8), dtype=wp.float32)):
  pass


class vec8b(wp.types.vector(length=8, dtype=wp.int8)):
  pass


class vec16b(wp.types.vector(length=16, dtype=wp.int8)):
  pass


@wp.func
def box_normals(i: int) -> wp.vec3:
  direction = wp.select(i < 3, 1.0, -1.0)
  mod = i % 3
  if mod == 0:
    return wp.vec3(0.0, direction, 0.0)
  if mod == 1:
    return wp.vec3(0.0, 0.0, direction)
  return wp.vec3(-direction, 0.0, 0.0)


@wp.struct
class Box:
  verts: mat83f


@wp.func
def box(R: wp.mat33, t: wp.vec3, geom_size: wp.vec3) -> Box:
  """Get a transformed box"""
  x = geom_size[0]
  y = geom_size[1]
  z = geom_size[2]
  t = wp.vec3(0.0)
  m = mat38f(
    R @ wp.vec3(-x, -y, -z) + t,
    R @ wp.vec3(-x, -y, +z) + t,
    R @ wp.vec3(-x, +y, -z) + t,
    R @ wp.vec3(-x, +y, +z) + t,
    R @ wp.vec3(+x, -y, -z) + t,
    R @ wp.vec3(+x, -y, +z) + t,
    R @ wp.vec3(+x, +y, -z) + t,
    R @ wp.vec3(+x, +y, +z) + t,
  )
  return Box(wp.transpose(m))


@wp.func
def get_box_axis(
  axis_idx: int,
  R: wp.mat33,
) -> tuple[wp.vec3, bool]:
  """Get the axis at index axis_idx.
  R: rotation matrix from a to b
  Axes 0-12 are face normals of boxes a & b
  Axes 12-21 are edge cross products."""
  if axis_idx < 6:  # a faces
    axis = R @ wp.vec3(box_normals(axis_idx))
    is_degenerate = False
  elif axis_idx < 12:  # b faces
    axis = wp.vec3(box_normals(axis_idx - 6))
    is_degenerate = False
  else:  # edges cross products
    assert axis_idx < 21
    edges = axis_idx - 12
    axis_a, axis_b = edges / 3, edges % 3
    edge_a = R[axis_a]
    if axis_b == 0:
      axis = wp.vec3(0.0, -edge_a[2], edge_a[1])
    elif axis_b == 1:
      axis = wp.vec3(edge_a[2], 0.0, -edge_a[0])
    else:
      axis = wp.vec3(-edge_a[1], edge_a[0], 0.0)
    is_degenerate = wp.length_sq(axis) < TINY_VAL
  return wp.normalize(axis), is_degenerate


@wp.func
def get_box_axis_support(
  axis: wp.vec3, degenerate_axis: bool, a: Box, b: Box
) -> tuple[wp.float32, wp.int32]:
  """Get the overlap (or separating distance if negative) along `axis`, and the sign."""
  axis_d = wp.vec3d(axis)
  support_a_max, support_b_max = wp.float32(-HUGE_VAL), wp.float32(-HUGE_VAL)
  support_a_min, support_b_min = wp.float32(HUGE_VAL), wp.float32(HUGE_VAL)
  for i in range(8):
    vert_a = wp.vec3d(a.verts[i])
    vert_b = wp.vec3d(b.verts[i])
    proj_a = wp.float32(wp.dot(vert_a, axis_d))
    proj_b = wp.float32(wp.dot(vert_b, axis_d))
    support_a_max = wp.max(support_a_max, proj_a)
    support_b_max = wp.max(support_b_max, proj_b)
    support_a_min = wp.min(support_a_min, proj_a)
    support_b_min = wp.min(support_b_min, proj_b)
  dist1 = support_a_max - support_b_min
  dist2 = support_b_max - support_a_min
  dist = wp.select(degenerate_axis, wp.min(dist1, dist2), HUGE_VAL)
  sign = wp.select(dist1 > dist2, 1, -1)
  return dist, sign


@wp.struct
class AxisSupport:
  best_dist: wp.float32
  best_sign: wp.int8
  best_idx: wp.int8


@wp.func
def reduce_axis_support(a: AxisSupport, b: AxisSupport):
  return wp.select(a.best_dist > b.best_dist, a, b)


@wp.func
def collision_axis_tiled(
  m: Model,
  d: Data,
  a: Box,
  b: Box,
  R: wp.mat33,
  axis_idx: wp.int32,
):
  """Finds the axis of minimum separation or maximum overlap.
  Returns:
    best_axis: vec3
    best_idx: int8
  """
  # launch tiled with block_dim=21

  axis, degenerate_axis = get_box_axis(axis_idx, R)
  axis_dist, axis_sign = get_box_axis_support(axis, degenerate_axis, a, b)

  supports = wp.tile(AxisSupport(axis_dist, wp.int8(axis_sign), wp.int8(axis_idx)))

  face_supports = wp.tile_view(supports, offset=(0,), shape=(12,))
  edge_supports = wp.tile_view(supports, offset=(12,), shape=(9,))

  face_supports_red = wp.tile_reduce(reduce_axis_support, face_supports)
  edge_supports_red = wp.tile_reduce(reduce_axis_support, edge_supports)

  face = face_supports_red[0]
  edge = edge_supports_red[0]

  if axis_idx > 0:
    return wp.vec3(0.0), 0

  face_axis, _1 = get_box_axis(wp.int32(face.best_idx), R)
  best_axis = wp.vec3(face_axis)

  if edge.best_dist < face.best_dist:
    edge_axis, _2 = get_box_axis(wp.int32(edge.best_idx), R)
    if wp.abs(wp.dot(face_axis, edge_axis)) < 0.99:
      best_axis = edge_axis

  # face_axis = get_axis(best_face_idx)[0]
  # best_axis = wp.vec3(face_axis)

  # if best_edge_dist < best_face_dist:
  #   edge_axis = get_axis(best_edge_idx)[0]
  #   if wp.abs(wp.dot(face_axis, edge_axis)) < 0.99:
  #     best_axis = edge_axis

  # # get the (reference) face most aligned with the separating axis
  return best_axis, 0


@wp.func
def create_contact_manifold(
  m: Model,
  d: Data,
) -> tuple[wp.float32, wp.vec3, wp.mat33]:
  pass


@wp.kernel
def box_box_kernel(
  m: Model,
  d: Data,
  num_kernels: int,
):
  """Calculates contacts between pairs of boxes."""
  thread_idx, axis_idx = wp.tid()

  key = wp.static(group_key(GeomType.BOX.value, GeomType.BOX.value))

  num_candidate_contacts = d.narrowphase_candidate_group_count[key]

  for bp_idx in range(thread_idx, num_candidate_contacts, num_kernels):
    geoms = d.narrowphase_candidate_geom[key, bp_idx]
    world_id = d.narrowphase_candidate_worldid[key, bp_idx]

    ga = geoms[0]
    gb = geoms[1]

    # transformations
    a_pos, b_pos = d.geom_xpos[world_id, ga], d.geom_xpos[world_id, gb]
    a_mat, b_mat = d.geom_xmat[world_id, ga], d.geom_xmat[world_id, gb]
    b_mat_inv = wp.transpose(b_mat)
    trans_atob = b_mat_inv @ (a_pos - b_pos)
    rot_atob = b_mat_inv @ a_mat

    a_size = m.geom_size[ga]
    b_size = m.geom_size[gb]
    a = box(rot_atob, trans_atob, a_size)
    b = box(wp.identity(3, wp.float32), wp.vec3(0.0), b_size)

    # - faces compute from verts

    # box-box implementation
    collision_axis_tiled(m, d, a, b, rot_atob, axis_idx)
    # dist, pos, normal = create_contact_manifold(m, d, axis_idx)

    # # generate contact w/ 1 thread per pair
    # if axis_idx != 0:
    #   return

    # # if contact
    # if dist < 0:
    #   contact_idx = wp.atomic_add(d.contact_counter, 0, 1)
    #   # TODO(ca): multi-dimensional contacts
    #   d.contact.dist[contact_idx] = dist
    #   d.contact.pos[contact_idx] = b_pos + b_mat @ pos
    #   d.contact.frame[contact_idx] = make_frame(b_mat @ normal)
    #   d.contact.worldid[contact_idx] = world_id


def box_box(
  m: Model,
  d: Data,
):
  """Calculates contacts between pairs of boxes."""
  kernel_ratio = 16
  num_kernels = math.ceil(
    d.nconmax / kernel_ratio
  )  # parallel kernels excluding tile dim
  wp.launch_tiled(
    kernel=box_box_kernel,
    dim=num_kernels,
    inputs=[m, d, num_kernels],
    block_dim=21,
  )


@wp.func
def _closest_segment_point_plane(
  a: wp.vec3, b: wp.vec3, p0: wp.vec3, plane_normal: wp.vec3
) -> wp.vec3:
  """Gets the closest point between a line segment and a plane.

  Args:
    a: first line segment point
    b: second line segment point
    p0: point on plane
    plane_normal: plane normal

  Returns:
    closest point between the line segment and the plane
  """
  # Parametrize a line segment as S(t) = a + t * (b - a), plug it into the plane
  # equation dot(n, S(t)) - d = 0, then solve for t to get the line-plane
  # intersection. We then clip t to be in [0, 1] to be on the line segment.
  n = plane_normal
  d = wp.dot(p0, n)  # shortest distance from origin to plane
  denom = wp.dot(n, (b - a))
  t = (d - wp.dot(n, a)) / (denom + TINY_VAL * wp.select(denom == 0.0, 1.0, 0.0))
  t = wp.clamp(t, 0.0, 1.0)
  segment_point = a + t * (b - a)

  return segment_point


@wp.kernel
def _clip_edge_to_poly(
  subject_poly: wp.array(dtype=wp.vec3, ndim=2),
  subject_poly_length: wp.array(dtype=wp.int32, ndim=1),
  clipping_poly: wp.array(dtype=wp.vec3, ndim=2),
  clipping_poly_length: wp.array(dtype=wp.int32, ndim=1),
  clipping_normal: wp.array(dtype=wp.vec3, ndim=1),
  clipped_points_offset: wp.array(dtype=wp.int32, ndim=1),
  # outputs
  clipped_points: wp.array(dtype=wp.vec3, ndim=2),
  mask: wp.array(dtype=wp.int32, ndim=2),
  clipped_points_length: wp.array(dtype=wp.int32, ndim=1),
):
  assert clipped_points.shape[1] == subject_poly.shape[1] + clipping_poly.shape[1]
  assert clipped_points.shape[1] == mask.shape[1]


@wp.func
def _project_quad_onto_plane(
  quad: mat43f,
  quad_n: wp.vec3,
  plane_n: wp.vec3,
  plane_pt: wp.vec3,
):
  """Projects poly1 onto the poly2 plane along poly2's normal."""
  d = wp.dot(plane_pt, plane_n)
  denom = wp.dot(quad_n, plane_n)
  qn_scaled = quad_n / (denom + wp.select(denom == 0.0, TINY_VAL, 0.0))

  for i in range(4):
    quad[i] = quad[i] + (d - wp.dot(quad[i], plane_n)) * qn_scaled
  return quad


@wp.func
def _project_poly_onto_plane(
  poly: Any,
  poly_n: wp.vec3,
  plane_n: wp.vec3,
  plane_pt: wp.vec3,
):
  """Projects poly1 onto the poly2 plane along poly2's normal."""
  d = wp.dot(plane_pt, plane_n)
  denom = wp.dot(poly_n, plane_n)
  qn_scaled = poly_n / (denom + wp.select(denom == 0.0, TINY_VAL, 0.0))

  for i in range(len(poly)):
    poly[i] = poly[i] + (d - wp.dot(poly[i], plane_n)) * qn_scaled
  return poly


@wp.func
def _clip_edge_to_quad(
  subject_poly: mat43f,
  clipping_poly: mat43f,
  clipping_normal: wp.vec3,
):
  p0 = mat43f()
  p1 = mat43f()
  mask = wp.vec4b()
  for edge_idx in range(4):
    subject_p0 = subject_poly[(edge_idx + 3) % 4]
    subject_p1 = subject_poly[edge_idx]

    any_both_in_front = wp.int32(0)
    clipped0_dist_max = wp.float32(-HUGE_VAL)
    clipped1_dist_max = wp.float32(-HUGE_VAL)
    clipped_p0_distmax = wp.vec3(0.0)
    clipped_p1_distmax = wp.vec3(0.0)

    for clipping_edge_idx in range(4):
      clipping_p0 = clipping_poly[(clipping_edge_idx + 3) % 4]
      clipping_p1 = clipping_poly[clipping_edge_idx]
      edge_normal = wp.cross(clipping_p1 - clipping_p0, clipping_normal)

      p0_in_front = wp.dot(subject_p0 - clipping_p0, edge_normal) > TINY_VAL
      p1_in_front = wp.dot(subject_p1 - clipping_p0, edge_normal) > TINY_VAL
      candidate_clipped_p = _closest_segment_point_plane(
        subject_p0, subject_p1, clipping_p1, edge_normal
      )
      clipped_p0 = wp.select(p0_in_front, subject_p0, candidate_clipped_p)
      clipped_p1 = wp.select(p1_in_front, subject_p1, candidate_clipped_p)
      clipped_dist_p0 = wp.dot(clipped_p0 - subject_p0, subject_p1 - subject_p0)
      clipped_dist_p1 = wp.dot(clipped_p1 - subject_p1, subject_p0 - subject_p1)
      any_both_in_front |= wp.int32(p0_in_front and p1_in_front)

      if clipped_dist_p0 > clipped0_dist_max:
        clipped0_dist_max = clipped_dist_p0
        clipped_p0_distmax = clipped_p0

      if clipped_dist_p1 > clipped1_dist_max:
        clipped1_dist_max = clipped_dist_p1
        clipped_p1_distmax = clipped_p1
    new_p0 = wp.select(any_both_in_front, clipped_p0_distmax, subject_p0)
    new_p1 = wp.select(any_both_in_front, clipped_p1_distmax, subject_p1)

    mask_val = wp.int8(
      wp.select(
        wp.dot(subject_p0 - subject_p1, new_p0 - new_p1) < 0,
        wp.int32(not any_both_in_front),
        0,
      )
    )

    p0[edge_idx] = new_p0
    p1[edge_idx] = new_p1
    mask[edge_idx] = mask_val
  return p0, p1, mask


@wp.func
def _clip_quad(
  subject_quad: mat43f,
  subject_normal: wp.vec3,
  clipping_quad: mat43f,
  clipping_normal: wp.vec3,
) -> tuple[mat83f, vec8b]:
  """Clips a subject quad against a clipping quad.
  Serial implementation.
  """

  subject_clipped_p0, subject_clipped_p1, subject_mask = _clip_edge_to_quad(
    subject_quad, clipping_quad, clipping_normal
  )
  clipping_proj = _project_poly_onto_plane(
    clipping_quad, clipping_normal, subject_normal, subject_quad[0]
  )
  clipping_clipped_p0, clipping_clipped_p1, clipping_mask = _clip_edge_to_quad(
    clipping_proj, subject_quad, subject_normal
  )

  clipped = mat16_3f()
  mask = vec16b()
  for i in range(4):
    clipped[i] = subject_clipped_p0[i]
    clipped[i + 4] = clipping_clipped_p0[i]
    clipped[i + 8] = subject_clipped_p1[i]
    clipped[i + 12] = clipping_clipped_p1[i]
    mask[i] = subject_mask[i]
    mask[i + 4] = clipping_mask[i]
    mask[i + 8] = subject_mask[i]
    mask[i + 8 + 4] = clipping_mask[i]

  return clipped, mask


# TODO(ca): sparse, tiling variant for large point counts
@wp.func
def _manifold_points(
  poly: Any,
  mask: Any,
  clipping_norm: wp.vec3,
) -> wp.vec4b:
  """Chooses four points on the polygon with approximately maximal area. Return the indices"""
  n = len(poly)

  a_idx = wp.int32(0)
  a_mask = wp.int8(mask[0])
  for i in range(n):
    if mask[i] >= a_mask:
      a_idx = i
      a_mask = mask[i]
  a = poly[a_idx]

  b_idx = wp.int32(0)
  b_dist = wp.float32(-HUGE_VAL)
  for i in range(n):
    dist = wp.length_sq(poly[i] - a) + wp.select(mask[i], -HUGE_VAL, 0.0)
    if dist >= b_dist:
      b_idx = i
      b_dist = dist
  b = poly[b_idx]

  ab = wp.cross(clipping_norm, a - b)

  c_idx = wp.int32(0)
  c_dist = wp.float32(-HUGE_VAL)
  for i in range(n):
    ap = a - poly[i]
    dist = wp.abs(wp.dot(ap, ab)) + wp.select(mask[i], -HUGE_VAL, 0.0)
    if dist >= c_dist:
      c_idx = i
      c_dist = dist
  c = poly[c_idx]

  ac = wp.cross(clipping_norm, a - c)
  bc = wp.cross(clipping_norm, b - c)

  d_idx = wp.int32(-2e6)
  d_dist = wp.float32(0)
  for i in range(n):
    ap = a - poly[i]
    dist_ap = wp.abs(wp.dot(ap, bc)) + wp.select(mask[i], -HUGE_VAL, 0.0)
    bp = b - poly[i]
    dist_bp = wp.abs(wp.dot(bp, bc)) + wp.select(mask[i], -HUGE_VAL, 0.0)
    if dist_ap + dist_bp >= d_dist:
      d_idx = i
      d_dist = dist_ap + dist_bp
  d = poly[d_idx]
  return wp.vec4b(wp.int8(a_idx), wp.int8(b_idx), wp.int8(c_idx), wp.int8(d_idx))


@wp.func
def _create_contact_manifold(
  clipping_quad: mat43f,
  clipping_normal: wp.vec3,
  subject_quad: mat43f,
  subject_normal: wp.vec3,
  sep_axis: wp.vec3,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  # Clip the subject (incident) face onto the clipping (reference) face.
  # The incident points are clipped points on the subject polygon.
  incident, mask = _clip_quad(
    subject_quad, subject_normal, clipping_quad, clipping_normal
  )

  clipping_normal_neg = -clipping_normal
  d = wp.dot(clipping_quad[0], clipping_normal_neg) + TINY_VAL

  for i in range(16):
    if wp.dot(incident[i], clipping_normal_neg) > d:
      mask[i] = wp.int8(0)

  ref = _project_poly_onto_plane(
    incident, clipping_normal, clipping_normal, clipping_quad[0]
  )

  # # Choose four contact points.
  best = _manifold_points(ref, mask, clipping_normal)
  contact_pts = mat43f()
  dist = wp.vec4f()

  for i in range(4):
    idx = wp.int32(best[i])
    contact_pt = ref[idx]
    contact_pts[i] = contact_pt
    penetration_dir = incident[idx] - contact_pt
    penetration = wp.dot(penetration_dir, clipping_normal_neg)
    dist[i] = wp.select(mask[idx], 1.0, -penetration)

  return dist, contact_pts


@wp.func
def distance_point_plane(plane_normal: wp.vec3, plane_pos: wp.vec3, point: wp.vec3):
  plane_normal = wp.normalize(plane_normal)
  dist = wp.dot(point - plane_pos, plane_normal)
  return dist, plane_pos - plane_normal * dist


@wp.func
def plane_box(
  plane: GeomPlane,
  box: GeomBox,
  worldid: int,
  d: Data,
  margin: float,
  geom_indices: wp.vec2i,
):
  contact_count = int(0)

  # Check all 8 corners of the box
  for i in range(8):
    corner = wp.vec3(box.size.x * 0.5, box.size.y * 0.5, box.size.z * 0.5)
    if i % 2 == 0:
      corner.x = -corner.x
    if (i // 2) % 2 == 0:
      corner.y = -corner.y
    if i < 4:
      corner.z = -corner.z

    corner_world = box.rot * (corner) + box.pos

    dist, pos = distance_point_plane(plane.normal, plane.pos, corner_world)

    if dist < 0.0:
      write_contact(
        d, dist, pos, make_frame(plane.normal), margin, geom_indices, worldid
      )
      contact_count += 1

    if contact_count >= 4:
      break


_collision_functions = {
  (GeomType.PLANE.value, GeomType.SPHERE.value): plane_sphere,
  (GeomType.SPHERE.value, GeomType.SPHERE.value): sphere_sphere,
  (GeomType.PLANE.value, GeomType.CAPSULE.value): plane_capsule,
  (GeomType.PLANE.value, GeomType.BOX.value): plane_box,
  (GeomType.CAPSULE.value, GeomType.CAPSULE.value): capsule_capsule,
}


def create_collision_function_kernel(type1, type2):
  key = group_key(type1, type2)

  @wp.kernel
  def _collision_function_kernel(
    m: Model,
    d: Data,
  ):
    tid = wp.tid()
    num_candidate_contacts = d.narrowphase_candidate_group_count[key]
    if tid >= num_candidate_contacts:
      return

    geoms = d.narrowphase_candidate_geom[key, tid]
    worldid = d.narrowphase_candidate_worldid[key, tid]

    g1 = geoms[0]
    g2 = geoms[1]

    geom1 = wp.static(get_info(type1))(
      g1,
      m,
      d.geom_xpos[worldid],
      d.geom_xmat[worldid],
    )
    geom2 = wp.static(get_info(type2))(
      g2,
      m,
      d.geom_xpos[worldid],
      d.geom_xmat[worldid],
    )

    margin = wp.max(m.geom_margin[g1], m.geom_margin[g2])

    wp.static(_collision_functions[(type1, type2)])(
      geom1, geom2, worldid, d, margin, geoms
    )

  return _collision_function_kernel


_collision_kernels = {}


def narrowphase(m: Model, d: Data):
  # we need to figure out how to keep the overhead of this small - not launching anything
  # for pair types without collisions, as well as updating the launch dimensions.

  # TODO only generate collision kernels we actually need
  if len(_collision_kernels) == 0:
    for type1, type2 in _collision_functions.keys():
      _collision_kernels[(type1, type2)] = create_collision_function_kernel(
        type1, type2
      )

  for collision_kernel in _collision_kernels.values():
    wp.launch(collision_kernel, dim=d.nconmax, inputs=[m, d])
