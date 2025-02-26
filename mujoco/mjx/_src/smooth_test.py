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

"""Tests for smooth dynamics functions."""

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
from mujoco import mjx
import numpy as np
import warp as wp

wp.config.verify_cuda = True

from . import test_util

# tolerance for difference between MuJoCo and mjWarp smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class SmoothTest(parameterized.TestCase):
  def test_kinematics(self):
    """Tests kinematics."""
    _, mjd, m, d = test_util.fixture("pendula.xml")

    for arr in (d.xanchor, d.xaxis, d.xquat, d.xpos):
      arr.zero_()

    mjx.kinematics(m, d)

    _assert_eq(d.xanchor.numpy()[0], mjd.xanchor, "xanchor")
    _assert_eq(d.xaxis.numpy()[0], mjd.xaxis, "xaxis")
    _assert_eq(d.xquat.numpy()[0], mjd.xquat, "xquat")
    _assert_eq(d.xpos.numpy()[0], mjd.xpos, "xpos")

  def test_com_pos(self):
    """Tests com_pos."""
    _, mjd, m, d = test_util.fixture("pendula.xml")

    for arr in (d.subtree_com, d.cinert, d.cdof):
      arr.zero_()

    mjx.com_pos(m, d)
    _assert_eq(d.subtree_com.numpy()[0], mjd.subtree_com, "subtree_com")
    _assert_eq(d.cinert.numpy()[0], mjd.cinert, "cinert")
    _assert_eq(d.cdof.numpy()[0], mjd.cdof, "cdof")

  def test_crb(self):
    """Tests crb."""
    _, mjd, m, d = test_util.fixture("pendula.xml")

    d.crb.zero_()

    mjx.crb(m, d)
    _assert_eq(d.crb.numpy()[0], mjd.crb, "crb")
    _assert_eq(d.qM.numpy()[0, 0], mjd.qM, "qM")

  def test_factor_m_sparse(self):
    """Tests factor_m (sparse)."""
    _, mjd, m, d = test_util.fixture("pendula.xml", sparse=True)

    for arr in (d.qLD, d.qLDiagInv):
      arr.zero_()

    mjx.factor_m(m, d)
    _assert_eq(d.qLD.numpy()[0, 0], mjd.qLD, "qLD (sparse)")
    _assert_eq(d.qLDiagInv.numpy()[0], mjd.qLDiagInv, "qLDiagInv")

  def test_factor_m_dense(self):
    """Tests MJX factor_m (dense)."""
    # TODO(team): switch this to pendula.xml and merge with above test
    # after mmacklin's tile_cholesky fixes are in
    _, mjd, m, d = test_util.fixture("humanoid/humanoid.xml", sparse=False)

    qLD = d.qLD.numpy()[0].copy()
    d.qLD.zero_()

    mjx.factor_m(m, d)
    _assert_eq(d.qLD.numpy()[0], qLD, "qLD (dense)")

  @parameterized.parameters(True, False)
  def test_solve_m(self, sparse: bool):
    """Tests solve_m."""
    # TODO(team): switch this to pendula.xml and merge with above test
    # after mmacklin's tile_cholesky fixes are in
    fname = "pendula.xml" if sparse else "humanoid/humanoid.xml"
    mjm, mjd, m, d = test_util.fixture(fname, sparse=sparse)

    qfrc_smooth = np.tile(mjd.qfrc_smooth, (1, 1))
    qacc_smooth = np.zeros(
      shape=(
        1,
        mjm.nv,
      ),
      dtype=float,
    )
    mujoco.mj_solveM(mjm, mjd, qacc_smooth, qfrc_smooth)

    d.qacc_smooth.zero_()

    mjx.solve_m(m, d, d.qacc_smooth, d.qfrc_smooth)
    _assert_eq(d.qacc_smooth.numpy()[0], qacc_smooth[0], "qacc_smooth")

  def test_rne(self):
    """Tests rne."""
    _, mjd, m, d = test_util.fixture("pendula.xml")

    d.qfrc_bias.zero_()

    mjx.rne(m, d)
    _assert_eq(d.qfrc_bias.numpy()[0], mjd.qfrc_bias, "qfrc_bias")

  def test_com_vel(self):
    """Tests com_vel."""
    _, mjd, m, d = test_util.fixture("pendula.xml")

    for arr in (d.cvel, d.cdof_dot):
      arr.zero_()

    mjx.com_vel(m, d)
    _assert_eq(d.cvel.numpy()[0], mjd.cvel, "cvel")
    _assert_eq(d.cdof_dot.numpy()[0], mjd.cdof_dot, "cdof_dot")

  def test_transmission(self):
    """Tests transmission."""
    mjm, mjd, m, d = test_util.fixture("pendula.xml")

    for arr in (d.actuator_length, d.actuator_moment):
      arr.zero_()

    actuator_moment = np.zeros((mjm.nu, mjm.nv))
    mujoco.mju_sparse2dense(
      actuator_moment,
      mjd.actuator_moment,
      mjd.moment_rownnz,
      mjd.moment_rowadr,
      mjd.moment_colind,
    )

    mjx._src.smooth.transmission(m, d)
    _assert_eq(d.actuator_length.numpy()[0], mjd.actuator_length, "actuator_length")
    _assert_eq(d.actuator_moment.numpy()[0], actuator_moment, "actuator_moment")


if __name__ == "__main__":
  wp.init()
  absltest.main()
