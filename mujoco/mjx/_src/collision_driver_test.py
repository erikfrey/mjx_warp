"""Tests for collision driver."""

from absl.testing import absltest
from mujoco import mjx
import numpy as np
import warp as wp

from . import test_util

# tolerance for difference between MuJoCo and MJX smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5

def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f'mismatch: {name}'
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class CollisionDriverTest(absltest.TestCase):

  def test_collision(self):
    """Tests collision."""
    _, mjd, m, d = test_util.fixture('pendula.xml')

    for arr in (d.qfrc_spring, d.qfrc_damper, d.qfrc_passive):
      arr.zero_()

    mjx.collision(m, d)




if __name__ == '__main__':
  wp.init()
  absltest.main()
