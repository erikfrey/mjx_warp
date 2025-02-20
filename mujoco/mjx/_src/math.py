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

from . import types


@wp.func
def mul_quat(u: wp.quat, v: wp.quat) -> wp.quat:
  return wp.quat(
    u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
    u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
    u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
    u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
  )


@wp.func
def rot_vec_quat(vec: wp.vec3, quat: wp.quat) -> wp.vec3:
  s, u = quat[0], wp.vec3(quat[1], quat[2], quat[3])
  r = 2.0 * (wp.dot(u, vec) * u) + (s * s - wp.dot(u, u)) * vec
  r = r + 2.0 * s * wp.cross(u, vec)
  return r


@wp.func
def axis_angle_to_quat(axis: wp.vec3, angle: wp.float32) -> wp.quat:
  s, c = wp.sin(angle * 0.5), wp.cos(angle * 0.5)
  axis = axis * s
  return wp.quat(c, axis[0], axis[1], axis[2])


@wp.func
def quat_to_mat(quat: wp.quat) -> wp.mat33:
  """Converts a quaternion into a 9-dimensional rotation matrix."""
  vec = wp.vec4(quat[0], quat[1], quat[2], quat[3])
  q = wp.outer(vec, vec)

  return wp.mat33(
    q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3],
    2.0 * (q[1, 2] - q[0, 3]),
    2.0 * (q[1, 3] + q[0, 2]),
    2.0 * (q[1, 2] + q[0, 3]),
    q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3],
    2.0 * (q[2, 3] - q[0, 1]),
    2.0 * (q[1, 3] - q[0, 2]),
    2.0 * (q[2, 3] + q[0, 1]),
    q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3],
  )


@wp.func
def inert_vec(i: types.vec10, v: wp.spatial_vector) -> wp.spatial_vector:
  """mju_mulInertVec: multiply 6D vector (rotation, translation) by 6D inertia matrix."""
  return wp.spatial_vector(
    i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5],
    i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5],
    i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4],
    i[8] * v[1] - i[7] * v[2] + i[9] * v[3],
    i[6] * v[2] - i[8] * v[0] + i[9] * v[4],
    i[7] * v[0] - i[6] * v[1] + i[9] * v[5],
  )


@wp.func
def motion_cross(u: wp.spatial_vector, v: wp.spatial_vector) -> wp.spatial_vector:
  """Cross product of two motions."""

  u0 = wp.vec3(u[0], u[1], u[2], dtype=wp.float32)
  u1 = wp.vec3(u[3], u[4], u[5], dtype=wp.float32)
  v0 = wp.vec3(v[0], v[1], v[2], dtype=wp.float32)
  v1 = wp.vec3(v[3], v[4], v[5], dtype=wp.float32)

  ang = wp.cross(u0, v0)
  vel = wp.cross(u1, v0) + wp.cross(u0, v1)

  return wp.spatial_vector(ang, vel)


@wp.func
def motion_cross_force(v: wp.spatial_vector, f: wp.spatial_vector) -> wp.spatial_vector:
  """Cross product of a motion and a force."""

  v0 = wp.vec3(v[0], v[1], v[2], dtype=wp.float32)
  v1 = wp.vec3(v[3], v[4], v[5], dtype=wp.float32)
  f0 = wp.vec3(f[0], f[1], f[2], dtype=wp.float32)
  f1 = wp.vec3(f[3], f[4], f[5], dtype=wp.float32)

  ang = wp.cross(v0, f0) + wp.cross(v1, f1)
  vel = wp.cross(v0, f1)

  return wp.spatial_vector(ang, vel)


@wp.func
def quat_to_vel(quat: wp.quat) -> wp.vec3:
  axis = wp.vec3(quat[1], quat[2], quat[3])
  sin_a_2 = wp.norm_l2(axis)

  if sin_a_2 == 0.0:
    return wp.vec3(0.0)

  speed = 2.0 * wp.atan2(sin_a_2, quat[0])
  # when axis-angle is larger than pi, rotation is in the opposite direction
  if speed > wp.pi:
    speed -= 2.0 * wp.pi

  return axis * speed / sin_a_2


@wp.func
def quat_sub(qa: wp.quat, qb: wp.quat) -> wp.vec3:
  """Subtract quaternions, express as 3D velocity: qb*quat(res) = qa."""
  # qdif = neg(qb)*qa
  qneg = wp.quat(qb[0], -qb[1], -qb[2], -qb[3])
  qdif = mul_quat(qneg, qa)

  # convert to 3D velocity
  return quat_to_vel(qdif)


@wp.func
def quat_integrate(q: wp.quat, v: wp.vec3, dt: wp.float32) -> wp.quat:
  """Integrates a quaternion given angular velocity and dt."""
  norm_ = wp.length(v)
  v = wp.normalize(v)  # does that need proper zero gradient handling?
  angle = dt * norm_

  q_res = axis_angle_to_quat(v, angle)
  q_res = mul_quat(q, q_res)

  return wp.normalize(q_res)
