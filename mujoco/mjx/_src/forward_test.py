"""Tests for forward dynamics functions."""

from absl.testing import absltest
from etils import epath
import mujoco
from mujoco import mjx
import numpy as np
import warp as wp

# tolerance for difference between MuJoCo and MJX smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f'mismatch: {name}'
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class ForwardTest(absltest.TestCase):

  def _load(self, fname: str, is_sparse: bool = True):
    path = epath.resource_path('mujoco.mjx') / 'test_data' / fname
    mjm = mujoco.MjModel.from_xml_path(path.as_posix())
    mjm.opt.jacobian = is_sparse
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 1)  # reset to stand_on_left_leg
    mjd.qvel = np.random.uniform(low=-0.01, high=0.01, size=mjd.qvel.shape)
    mujoco.mj_forward(mjm, mjd)
    m = mjx.put_model(mjm)
    d = mjx.put_data(mjm, mjd)
    return mjm, mjd, m, d

  def test_fwd_velocity(self):
    """Tests MJX fwd_velocity."""
    _, mjd, m, d = self._load('humanoid/humanoid.xml')
    
    d.actuator_velocity.zero_()
    mjx.fwd_velocity(m, d)

    _assert_eq(d.actuator_velocity.numpy()[0], mjd.actuator_velocity, 'actuator_velocity')
    _assert_eq(d.qfrc_bias.numpy()[0], mjd.qfrc_bias, 'qfrc_bias')

  def test_fwd_acceleration(self):
    """Tests MJX fwd_acceleration."""
    _, mjd, m, d = self._load('humanoid/humanoid.xml', is_sparse=False)

    for arr in (d.qfrc_smooth, d.qacc_smooth):
      arr.zero_()

    mjx.factor_m(m, d) # for dense, get tile cholesky factorization
    mjx.fwd_acceleration(m, d)

    _assert_eq(d.qfrc_smooth.numpy()[0], mjd.qfrc_smooth, 'qfrc_smooth')
    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, 'qacc_smooth')


if __name__ == '__main__':
  wp.init()
  absltest.main()
