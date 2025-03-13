import itertools

import numpy as np
import warp as wp
from absl.testing import absltest

from .collision_functions import collision_axis_tiled, Box


@wp.kernel
def test_collision_axis_tiled_kernel(
  a: Box,
  b: Box,
  R: wp.mat33,
  best_axis: wp.array(dtype=wp.vec3),
  best_sign: wp.array(dtype=wp.int32),
  best_idx: wp.array(dtype=wp.int32),
):
  bid, axis_idx = wp.tid()
  axis_out, sign_out, idx_out = collision_axis_tiled(a, b, R, axis_idx)
  if axis_idx > 0:
    return
  best_axis[bid] = axis_out
  best_sign[bid] = sign_out
  best_idx[bid] = idx_out


class TestCollisionAxisTiled(absltest.TestCase):
  """Tests the collision_axis_tiled function."""

  def test_collision_axis_tiled_vf(self):
    """Tests the collision_axis_tiled function."""
    vert = np.array(list(itertools.product((-1, 1), (-1, 1), (-1, 1))), dtype=float)

    dims_a = np.array([1.1, 0.8, 1.8])
    dims_b = np.array([0.8, 0.3, 1.1])

    # shift in yz plane
    shift_b = np.array([0, 0.2, -0.4])

    separation = 0.13

    s = 0.5 * 2**0.5
    rx = np.array([1, 0, 0, 0, s, s, 0, -s, s]).reshape((3, 3))
    ry = np.array([s, 0, s, 0, 1, 0, -s, 0, s]).reshape((3, 3))
    # Rotate vert 0 towards negative x at origin
    R_a = ry @ rx.T
    t_a = (-dims_a * vert[0]) @ R_a.T

    R_b = np.eye(3)
    t_b = -np.array([dims_b[0] + separation, 0, 0]) + shift_b

    vert_a = (dims_a * vert) @ R_a.T + t_a
    vert_b = dims_b * vert + t_b

    R_atob = R_b.T @ R_a
    t_atob = R_b.T @ (t_a - t_b)
    a = Box(vert_a.ravel())
    b = Box(vert_b.ravel())

    R = wp.mat33(R_atob)
    best_axis = wp.empty(1, dtype=wp.vec3)
    best_sign = wp.empty(1, dtype=wp.int32)
    best_idx = wp.empty(1, dtype=wp.int32)

    wp.launch_tiled(
      kernel=test_collision_axis_tiled_kernel,
      dim=1,
      inputs=[a, b, R],
      outputs=[best_axis, best_sign, best_idx],
      block_dim=256,
    )
    expected_axis = np.array([-1, 0, 0])
    expected_sign = np.sign(best_axis.numpy().dot(expected_axis))

    np.testing.assert_array_equal(
      np.cross(best_axis.numpy()[0], expected_axis), np.zeros(3)
    )
    # np.testing.assert_array_equal(best_idx.numpy()[0], 8)
    np.testing.assert_array_equal(best_sign.numpy()[0], expected_sign)

  def test_collision_axis_tiled_ee(self):
    """Tests the collision_axis_tiled function."""
    # edge on edge
    vert = np.array(list(itertools.product((-1, 1), (-1, 1), (-1, 1))), dtype=float)

    dims_a = np.array([1.1, 0.8, 1.8])
    dims_b = np.array([0.8, 0.3, 1.1])

    dims_a = dims_b = np.ones(3)

    # shift

    separation = 0.13

    s = 0.5 * 2**0.5
    rx = np.array([1, 0, 0, 0, s, s, 0, -s, s]).reshape((3, 3))
    ry = np.array([s, 0, s, 0, 1, 0, -s, 0, s]).reshape((3, 3))

    # Rotate vert 0 towards negative x at origin
    R_a = ry @ rx
    t_a = (-dims_a * 0.5 * (vert[2] + vert[6])) @ R_a.T
    # t_a = np.array([0, 0, 2**0.5])

    R_b = np.eye(3)
    t_b = (-dims_b * vert[0]) @ R_b.T
    t_b = np.array([-1, 0.0, -1])
    t_b -= np.array([s, 0, s]) * 0.2

    vert_a = (dims_a * vert) @ R_a.T + t_a
    vert_b = dims_b * vert @ R_b.T + t_b

    R_atob = R_b.T @ R_a
    t_atob = R_b.T @ (t_a - t_b)
    a = Box(vert_a.ravel())
    b = Box(vert_b.ravel())

    R = wp.mat33(R_atob)
    best_axis = wp.empty(1, dtype=wp.vec3)
    best_sign = wp.empty(1, dtype=wp.int32)
    best_idx = wp.empty(1, dtype=wp.int32)

    wp.launch_tiled(
      kernel=test_collision_axis_tiled_kernel,
      dim=1,
      inputs=[a, b, R],
      outputs=[best_axis, best_sign, best_idx],
      block_dim=256,
    )
    expected_axis = np.array([-1, 0, -1])
    expected_sign = np.sign(best_axis.numpy().dot(expected_axis))

    np.testing.assert_array_equal(
      np.cross(best_axis.numpy()[0], expected_axis), np.zeros(3)
    )
    # np.testing.assert_array_equal(best_idx.numpy()[0], 8)
    np.testing.assert_array_equal(best_sign.numpy()[0], expected_sign)


if __name__ == "__main__":
  absltest.main()
