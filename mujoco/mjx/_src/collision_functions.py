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
from .math import matmul_unroll_33
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
def _clearance_gradient(
  geom1_pos: wp.vec3,
  geom2_pos: wp.vec3,
  geom1_mat: wp.mat33,
  geom2_mat: wp.mat33,
  geom1_radius: wp.float32,
  geom2_radius: wp.float32,
  geom1_halfsize: wp.float32,
  geom2_halfsize: wp.float32,
  pos: wp.vec3,
) -> wp.vec3:
  relmat = matmul_unroll_33(wp.transpose(geom1_mat), geom2_mat)
  relpos = wp.transpose(geom1_mat) @ (geom2_pos - geom1_pos)
  new_pos = relmat @ pos + relpos

  cylinder1_pos = _cylinder(new_pos, geom1_radius, geom1_halfsize)
  cylinder2_pos = _cylinder(pos, geom2_radius, geom2_halfsize)

  grad_cylinder1 = _cylinder_gradient(new_pos, geom1_radius, geom1_halfsize)
  grad_cylinder1 = relmat @ grad_cylinder1
  grad_cylinder2 = _cylinder_gradient(pos, geom2_radius, geom2_halfsize)

  grad_temp = wp.select(cylinder1_pos > cylinder2_pos, grad_cylinder2, grad_cylinder1)
  sca = wp.select(wp.max(cylinder1_pos, cylinder2_pos) > 0.0, -1.0, 1.0)
  gradient = wp.vec3(
    grad_cylinder1[0] + grad_cylinder2[0] + grad_temp[0] * sca,
    grad_cylinder1[1] + grad_cylinder2[1] + grad_temp[1] * sca,
    grad_cylinder1[2] + grad_cylinder2[2] + grad_temp[2] * sca,
  )

  return gradient


@wp.func
def _cylinder_gradient(
  pos: wp.vec3, radius: wp.float32, halfsize: wp.float32
) -> wp.vec3:
  c = wp.sqrt(pos[0] * pos[0] + pos[1] * pos[1])
  e = wp.abs(pos[2])
  a0 = c - radius
  a1 = e - halfsize
  gradient = wp.vec3(0.0)
  amax = wp.max(a0, a1)

  if amax < 0.0:
    if a0 < amax:
      gradient[2] = wp.select(e == 0.0, pos[2] / e, 0.0)

    if a1 < amax:
      gradient[0] = wp.select(c == 0.0, pos[0] / c, 0.0)
      gradient[1] = wp.select(c == 0.0, pos[1] / c, 0.0)

  else:
    b0 = wp.max(a0, 0.0)
    b1 = wp.max(a1, 0.0)

    bnorm = wp.sqrt(b0 * b0 + b1 * b1)
    gradient[0] = (b0 / bnorm) * (pos[0] / c)
    gradient[1] = (b0 / bnorm) * (pos[1] / c)
    gradient[2] = (b1 / bnorm) * (pos[2] / e)

    if gradient[0] != gradient[0]:
      # Handling special cases inducing NaN values
      if bnorm == 0.0 and c == 0.0:
        gradient[0] = 0.0
        gradient[1] = 0.0
      elif bnorm == 0.0:
        gradient[0] = pos[0] / c
        gradient[1] = pos[1] / c
      elif c == 0.0:
        gradient[0] = b0 / bnorm
        gradient[1] = b0 / bnorm

    if gradient[2] != gradient[2]:
      # Handling special cases inducing NaN values
      if bnorm == 0.0 and e == 0.0:
        gradient[2] = 0.0
      elif bnorm == 0.0:
        gradient[2] = pos[2] / e
      elif e == 0.0:
        gradient[2] = b1 / bnorm

  return gradient


@wp.func
def _cylinder(pos: wp.vec3, radius: wp.float32, halfsize: wp.float32) -> wp.float32:
  a0 = wp.sqrt(pos[0] * pos[0] + pos[1] * pos[1]) - radius
  a1 = wp.abs(pos[2]) - halfsize
  b0 = wp.max(a0, 0.0)
  b1 = wp.max(a1, 0.0)
  return wp.min(wp.max(a0, a1), 0.0) + wp.sqrt(b0 * b0 + b1 * b1)


@wp.func
def _cylinder_frame(
  pos: wp.vec3,
  from_pos: wp.vec3,
  from_mat: wp.mat33,
  to_pos: wp.vec3,
  to_mat: wp.mat33,
  radius: wp.float32,
  halfsize: wp.float32,
) -> wp.float32:
  relmat = matmul_unroll_33(wp.transpose(to_mat), from_mat)
  relpos = wp.transpose(to_mat) @ (from_pos - to_pos)
  new_pos = relmat @ pos + relpos

  return _cylinder(new_pos, radius, halfsize)


@wp.func
def _gradient_step(
  geom1_pos: wp.vec3,
  geom2_pos: wp.vec3,
  geom1_mat: wp.mat33,
  geom2_mat: wp.mat33,
  geom1_radius: wp.float32,
  geom2_radius: wp.float32,
  geom1_halfsize: wp.float32,
  geom2_halfsize: wp.float32,
  x: wp.vec3,
):
  """Performs a step of gradient descent."""
  amin = 1.0e-4  # minimum value for line search factor scaling the gradient
  amax = 2.0  # maximum value for line search factor scaling the gradient
  nlinesearch = 10  # line search points

  grad_clearance = _clearance_gradient(
    geom1_pos,
    geom2_pos,
    geom1_mat,
    geom2_mat,
    geom1_radius,
    geom2_radius,
    geom1_halfsize,
    geom2_halfsize,
    x,
  )

  ratio = (amax / amin) ** (1.0 / float(nlinesearch - 1))
  value_prev = 1.0e10
  candidate_prev = wp.vec3(0.0)
  for i in range(nlinesearch):
    alpha = amin * (ratio ** float(i))
    candidate = wp.vec3(
      x[0] - alpha * grad_clearance[0],
      x[1] - alpha * grad_clearance[1],
      x[2] - alpha * grad_clearance[2],
    )
    cylinder1_pos = _cylinder_frame(
      candidate,
      geom2_pos,
      geom2_mat,
      geom1_pos,
      geom1_mat,
      geom1_radius,
      geom1_halfsize,
    )
    cylinder2_pos = _cylinder(candidate, geom2_radius, geom2_halfsize)
    value = cylinder1_pos + cylinder2_pos + wp.abs(wp.max(cylinder1_pos, cylinder2_pos))
    if value < value_prev:
      value_prev = value
      candidate_prev = candidate
    else:
      return candidate_prev

  return candidate


@wp.func
def _gradient_descent(
  geom1_pos: wp.vec3,
  geom2_pos: wp.vec3,
  geom1_mat: wp.mat33,
  geom2_mat: wp.mat33,
  geom1_radius: wp.float32,
  geom2_radius: wp.float32,
  geom1_halfsize: wp.float32,
  geom2_halfsize: wp.float32,
  x: wp.vec3,
  niter: int,
):
  for _ in range(niter):
    x = _gradient_step(
      geom1_pos,
      geom2_pos,
      geom1_mat,
      geom2_mat,
      geom1_radius,
      geom2_radius,
      geom1_halfsize,
      geom2_halfsize,
      x,
    )

  return x


@wp.func
def _optim(
  geom1_pos: wp.vec3,
  geom2_pos: wp.vec3,
  geom1_mat: wp.mat33,
  geom2_mat: wp.mat33,
  geom1_radius: wp.float32,
  geom2_radius: wp.float32,
  geom1_halfsize: wp.float32,
  geom2_halfsize: wp.float32,
  x0: wp.vec3,
):
  """Optimizes the clearance function."""
  print(geom2_mat)
  x0 = wp.transpose(geom2_mat) @ (x0 - geom2_pos)
  pos = _gradient_descent(
    geom1_pos,
    geom2_pos,
    geom1_mat,
    geom2_mat,
    geom1_radius,
    geom2_radius,
    geom1_halfsize,
    geom2_halfsize,
    x0,
    10,
  )
  cylinder1_pos = _cylinder_frame(
    pos, geom2_pos, geom2_mat, geom1_pos, geom1_mat, geom1_radius, geom1_halfsize
  )
  cylinder2_pos = _cylinder(pos, geom2_radius, geom2_halfsize)
  dist = cylinder1_pos + cylinder2_pos

  grad_cylinder1 = _cylinder_gradient(pos, geom1_radius, geom1_halfsize)
  grad_cylinder2 = _cylinder_gradient(pos, geom2_radius, geom2_halfsize)
  relmat = matmul_unroll_33(wp.transpose(geom1_mat), geom2_mat)
  grad_cylinder1 = relmat @ grad_cylinder1

  pos = geom2_mat @ pos + geom2_pos  # d2 to global frame
  n = wp.normalize(
    wp.vec3(
      grad_cylinder1[0] - grad_cylinder2[0],
      grad_cylinder1[1] - grad_cylinder2[1],
      grad_cylinder1[2] - grad_cylinder2[2],
    )
  )
  n = geom2_mat @ n
  return dist, pos, make_frame(n)


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


@wp.func
def cylinder_cylinder(
  cylinder1: GeomCylinder, cylinder2: GeomCylinder, worldid: int, d: Data
):
  """Calculates contacts between a cylinder and a cylinder object."""

  geom1_pos = cylinder1.pos
  geom2_pos = cylinder2.pos
  geom1_mat = cylinder1.rot
  geom2_mat = cylinder2.rot
  geom1_radius = cylinder1.radius
  geom2_radius = cylinder2.radius
  geom1_halfsize = cylinder1.halfsize
  geom2_halfsize = cylinder2.halfsize

  basis = make_frame(geom2_pos - cylinder1.pos)
  mid = 0.5 * (geom1_pos + geom2_pos)

  start_index = wp.atomic_add(d.ncon, 0, 4)
  index = start_index
  for condim in range(4):
    r = wp.max(cylinder1.radius, cylinder2.radius) * wp.select(condim < 2, -1.0, 1.0)
    condim_vector = wp.select(condim == 0 or condim == 2, basis[2], basis[1])
    x0 = mid + r * condim_vector
    dist, pos, frame = _optim(
      geom1_pos,
      geom2_pos,
      geom1_mat,
      geom2_mat,
      geom1_radius,
      geom2_radius,
      geom1_halfsize,
      geom2_halfsize,
      x0,
    )
    d.contact.dist[index] = dist
    d.contact.pos[index] = pos
    d.contact.frame[index] = frame
    d.contact.worldid[index] = worldid
    index += 1

  return start_index, 4


_collision_functions = {
  (GeomType.PLANE.value, GeomType.SPHERE.value): plane_sphere,
  (GeomType.SPHERE.value, GeomType.SPHERE.value): sphere_sphere,
  (GeomType.PLANE.value, GeomType.CAPSULE.value): plane_capsule,
  (GeomType.CYLINDER.value, GeomType.CYLINDER.value): cylinder_cylinder,
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
