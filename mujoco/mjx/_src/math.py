import warp as wp


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
def inert_vec(i: wp.types.vector(length=10, dtype=wp.float32), v: wp.spatial_vector) -> wp.spatial_vector:
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
def motion_cross_force(v: wp.spatial_vector, f: wp.spatial_vector) -> wp.spatial_vector:
  """Cross product of a motion and a force."""

  v0 = wp.vec3(v[0], v[1], v[2], dtype=wp.float32)
  v1 = wp.vec3(v[3], v[4], v[5], dtype=wp.float32)
  f0 = wp.vec3(f[0], f[1], f[2], dtype=wp.float32)
  f1 = wp.vec3(f[3], f[4], f[5], dtype=wp.float32)

  ang = wp.cross(v0, f0) + wp.cross(v1, f1)
  vel = wp.cross(v0, f1)

  return wp.spatial_vector(ang, vel)
