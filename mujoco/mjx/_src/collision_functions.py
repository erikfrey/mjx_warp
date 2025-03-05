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
from .types import GeomType
from .math import make_frame
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
def _plane_sphere(
  plane_normal: wp.vec3, plane_pos: wp.vec3, sphere_pos: wp.vec3, sphere_radius: float
):
  dist = wp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius
  pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
  return dist, pos


@wp.func
def plane_sphere(plane: GeomPlane, sphere: GeomSphere, worldid: int, d: Data):
  dist, pos = _plane_sphere(plane.normal, plane.pos, sphere.pos, sphere.radius)

  index = wp.atomic_add(d.ncon, 0, 1)
  d.contact.dist[index] = dist
  d.contact.pos[index] = pos
  d.contact.frame[index] = make_frame(plane.normal)
  return index, 1


@wp.func
def sphere_sphere(sphere1: GeomSphere, sphere2: GeomSphere, worldid: int, d: Data):
  dir = sphere1.pos - sphere2.pos
  dist = wp.length(dir)
  if dist == 0.0:
    n = wp.vec3(1.0, 0.0, 0.0)
  else:
    n = dir / dist
  dist = dist - (sphere1.radius + sphere2.radius)
  pos = sphere1.pos + n * (sphere1.radius + 0.5 * dist)

  index = wp.atomic_add(d.ncon, 0, 1)
  d.contact.dist[index] = dist
  d.contact.pos[index] = pos
  d.contact.frame[index] = make_frame(n)
  return index, 1


@wp.func
def plane_capsule(plane: GeomPlane, cap: GeomCapsule, worldid: int, d: Data):
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

  start_index = wp.atomic_add(d.ncon, 0, 2)
  index = start_index
  dist, pos = _plane_sphere(n, plane.pos, cap.pos + segment, cap.radius)
  d.contact.dist[index] = dist
  d.contact.pos[index] = pos
  d.contact.frame[index] = frame
  index += 1

  dist, pos = _plane_sphere(n, plane.pos, cap.pos - segment, cap.radius)
  d.contact.dist[index] = dist
  d.contact.pos[index] = pos
  d.contact.frame[index] = frame
  return start_index, 2


_collision_functions = {
  (GeomType.PLANE.value, GeomType.SPHERE.value): plane_sphere,
  (GeomType.SPHERE.value, GeomType.SPHERE.value): sphere_sphere,
  (GeomType.PLANE.value, GeomType.CAPSULE.value): plane_capsule,
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

    index, ncon = wp.static(_collision_functions[(type1, type2)])(
      geom1, geom2, worldid, d
    )
    for i in range(ncon):
      d.contact.worldid[index + i] = worldid
      d.contact.geom[index + i] = geoms

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
