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
