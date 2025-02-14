"""Tests for smooth dynamics functions."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
import scipy as sp
import warp as wp

import mujoco
from mujoco import mjx

# tolerance for difference between MuJoCo and MJX smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f'mismatch: {name}'
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class SmoothTest(parameterized.TestCase):

  def _load(self, fname: str, is_sparse: bool = True):
    path = epath.resource_path('mujoco.mjx') / 'test_data' / fname
    mjm = mujoco.MjModel.from_xml_path(path.as_posix())
    mjm.opt.jacobian = is_sparse
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 1) # reset to stand_on_left_leg
    mjd.qvel = np.random.uniform(low=-0.01, high=0.01, size=mjd.qvel.shape)
    mujoco.mj_forward(mjm, mjd)
    m = mjx.put_model(mjm)
    d = mjx.put_data(mjm, mjd)
    return mjm, mjd, m, d

  @parameterized.parameters('humanoid/humanoid.xml', 'humanoid/n_humanoids.xml')
  def test_kinematics(self, fname: str):
    """Tests MJX kinematics."""
    _, mjd, m, d = self._load(fname)

    for arr in (d.xanchor, d.xaxis, d.xquat, d.xpos):
      arr.zero_()

    mjx.kinematics(m, d)

    _assert_eq(d.xanchor.numpy()[0], mjd.xanchor, 'xanchor')
    _assert_eq(d.xaxis.numpy()[0], mjd.xaxis, 'xaxis')
    _assert_eq(d.xquat.numpy()[0], mjd.xquat, 'xquat')
    _assert_eq(d.xpos.numpy()[0], mjd.xpos, 'xpos')

  @parameterized.parameters('humanoid/humanoid.xml', 'humanoid/n_humanoids.xml')
  def test_com_pos(self, fname: str):
    """Tests MJX com_pos."""
    _, mjd, m, d = self._load(fname)

    for arr in (d.subtree_com, d.cinert, d.cdof):
      arr.zero_()

    mjx.com_pos(m, d)
    _assert_eq(d.subtree_com.numpy()[0], mjd.subtree_com, 'subtree_com')
    _assert_eq(d.cinert.numpy()[0], mjd.cinert, 'cinert')
    _assert_eq(d.cdof.numpy()[0], mjd.cdof, 'cdof')

  @parameterized.parameters('humanoid/humanoid.xml', 'humanoid/n_humanoids.xml')
  def test_crb(self, fname: str):
    """Tests MJX crb."""
    _, mjd, m, d = self._load(fname)

    for arr in (d.crb,):
      arr.zero_()

    mjx.crb(m, d)
    _assert_eq(d.crb.numpy()[0], mjd.crb, 'crb')
    _assert_eq(d.qM.numpy()[0, 0], mjd.qM, 'qM')

  @parameterized.parameters('humanoid/humanoid.xml', 'humanoid/n_humanoids.xml')
  def test_factor_m(self, fname: str):
    """Tests MJX factor_m."""
    _, mjd, m, d = self._load(fname)

    for arr in (d.qLD, d.qLDiagInv):
      arr.zero_()

    mjx.factor_m(m, d)
    _assert_eq(d.qLD.numpy()[0, 0], mjd.qLD, 'qLD (sparse)')
    _assert_eq(d.qLDiagInv.numpy()[0], mjd.qLDiagInv, 'qLDiagInv')

  @parameterized.parameters('humanoid/humanoid.xml', 'humanoid/n_humanoids.xml')
  def test_factor_m_dense(self, fname):
    """Tests MJX factor_m (dense)."""
    _, _, m, d = self._load(fname, is_sparse=False)

    qLD = d.qLD.numpy()[0].copy()
    d.qLD.zero_()

    mjx.factor_m(m, d)
    _assert_eq(d.qLD.numpy()[0].T, qLD, 'qLD (dense)')

  @parameterized.parameters('humanoid/humanoid.xml', 'humanoid/n_humanoids.xml')
  def test_rne(self, fname):
    """Tests MJX rne."""
    _, mjd, m, d = self._load(fname, is_sparse=False)

    d.qfrc_bias.zero_()

    mjx.rne(m, d)
    _assert_eq(d.qfrc_bias.numpy()[0], mjd.qfrc_bias, 'qfrc_bias')

  @parameterized.parameters('humanoid/humanoid.xml', 'humanoid/n_humanoids.xml')
  def test_solve_m_sparse(self, fname):
    """Tests solveM (sparse)"""
    mjm, mjd, m, d = self._load(fname)

    # zero the factorization
    mujoco.mju_zero(mjd.qLD)
    d.qLD.zero_()

    # re-run the factorization
    mujoco.mj_factorM(mjm, mjd)
    mjx.factor_m(m, d)

    _assert_eq(d.qLD.numpy()[0, 0], mjd.qLD, 'qLD (sparse)')

    # zero the output
    d.qacc_smooth.zero_()
    mujoco.mju_zero(mjd.qacc_smooth)

    # run the solve
    d.qacc_smooth = mjx.solve_m(m, d, d.qfrc_smooth, d.qacc_smooth)
    mujoco.mj_solveM(mjm, mjd, mjd.qacc_smooth.reshape(1, mjm.nv), mjd.qfrc_smooth.reshape(1, mjm.nv)) # why is the order of arguments different here?

    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, 'qacc_smooth (sparse)')

  @parameterized.parameters('humanoid/humanoid.xml', 'humanoid/n_humanoids.xml')
  def test_solve_m_dense(self, fname):
    """Tests solveM (sparse)"""
    mjm, mjd, m, d = self._load(fname, is_sparse=False)

    # construct dense M for comparison
    qM = np.zeros((mjm.nv, mjm.nv))
    mujoco.mj_fullM(mjm, qM, mjd.qM)
    _assert_eq(d.qM.numpy()[0], qM, 'qM (dense)')

    # cholesky factor for both
    qLD = np.linalg.cholesky(qM, upper=True)
    mjx.factor_m(m, d)

    # sanity comparison
    _assert_eq(d.qLD.numpy()[0].T, qLD, 'qLD (dense)')

    # zero the output
    d.qacc_smooth.zero_()
    mujoco.mju_zero(mjd.qacc_smooth)

    # solve
    d.qacc_smooth = mjx.solve_m(m, d, d.qfrc_smooth, d.qacc_smooth)
    mjd.qacc_smooth = sp.linalg.cho_solve((qLD, False), mjd.qfrc_smooth)
    
    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, 'qacc_smooth (dense)')

if __name__ == '__main__':
  wp.init()
  absltest.main()
