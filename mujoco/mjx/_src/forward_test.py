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

"""Tests for forward dynamics functions."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath

import mujoco
from mujoco import mjx

wp.config.verify_cuda = True

# tolerance for difference between MuJoCo and MJX smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class ForwardTest(absltest.TestCase):
  def _load(self, fname: str, is_sparse: bool = True):
    path = epath.resource_path("mujoco.mjx") / "test_data" / fname
    mjm = mujoco.MjModel.from_xml_path(path.as_posix())
    mjm.opt.jacobian = is_sparse
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 1)  # reset to stand_on_left_leg
    mjd.qvel = np.random.uniform(low=-0.01, high=0.01, size=mjd.qvel.shape)
    mjd.ctrl = np.random.normal(scale=10, size=mjd.ctrl.shape)
    mjd.act = np.random.normal(scale=10, size=mjd.act.shape)
    mujoco.mj_forward(mjm, mjd)
    m = mjx.put_model(mjm)
    d = mjx.put_data(mjm, mjd)
    return mjm, mjd, m, d

  def test_fwd_velocity(self):
    """Tests MJX fwd_velocity."""
    _, mjd, m, d = self._load("humanoid/humanoid.xml")

    d.actuator_velocity.zero_()
    mjx.fwd_velocity(m, d)

    _assert_eq(
      d.actuator_velocity.numpy()[0], mjd.actuator_velocity, "actuator_velocity"
    )
    _assert_eq(d.qfrc_bias.numpy()[0], mjd.qfrc_bias, "qfrc_bias")

  def test_fwd_actuation(self):
    """Tests MJX fwd_actuation."""
    mjm, mjd, m, d = self._load("humanoid/humanoid.xml", is_sparse=False)

    mujoco.mj_fwdActuation(mjm, mjd)

    for arr in (d.actuator_force, d.qfrc_actuator):
      arr.zero_()

    mjx.fwd_actuation(m, d)

    _assert_eq(d.ctrl.numpy()[0], mjd.ctrl, "ctrl")
    _assert_eq(d.actuator_force.numpy()[0], mjd.actuator_force, "actuator_force")
    _assert_eq(d.qfrc_actuator.numpy()[0], mjd.qfrc_actuator, "qfrc_actuator")

  def test_fwd_acceleration(self):
    """Tests MJX fwd_acceleration."""
    _, mjd, m, d = self._load("humanoid/humanoid.xml", is_sparse=False)

    for arr in (d.qfrc_smooth, d.qacc_smooth):
      arr.zero_()

    mjx.factor_m(m, d)  # for dense, get tile cholesky factorization
    mjx.fwd_acceleration(m, d)

    _assert_eq(d.qfrc_smooth.numpy()[0], mjd.qfrc_smooth, "qfrc_smooth")
    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, "qacc_smooth")

  def test_eulerdamp(self):
    path = epath.resource_path("mujoco.mjx") / "test_data/pendula.xml"
    mjm = mujoco.MjModel.from_xml_path(path.as_posix())
    self.assertTrue((mjm.dof_damping > 0).any())

    mjd = mujoco.MjData(mjm)
    mjd.qvel[:] = 1.0
    mjd.qacc[:] = 1.0
    mujoco.mj_forward(mjm, mjd)

    m = mjx.put_model(mjm)
    d = mjx.put_data(mjm, mjd)

    mjx.euler(m, d)
    mujoco.mj_Euler(mjm, mjd)

    _assert_eq(d.qpos.numpy()[0], mjd.qpos, "qpos")
    _assert_eq(d.act.numpy()[0], mjd.act, "act")

    # also test sparse
    mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    mjd = mujoco.MjData(mjm)
    mjd.qvel[:] = 1.0
    mjd.qacc[:] = 1.0
    mujoco.mj_forward(mjm, mjd)

    m = mjx.put_model(mjm)
    d = mjx.put_data(mjm, mjd)

    mjx.euler(m, d)
    mujoco.mj_Euler(mjm, mjd)

    _assert_eq(d.qpos.numpy()[0], mjd.qpos, "qpos")
    _assert_eq(d.act.numpy()[0], mjd.act, "act")

  def test_disable_eulerdamp(self):
    path = epath.resource_path("mujoco.mjx") / "test_data/pendula.xml"
    mjm = mujoco.MjModel.from_xml_path(path.as_posix())
    mjm.opt.disableflags = mjm.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_EULERDAMP

    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    mjd.qvel[:] = 1.0
    mjd.qacc[:] = 1.0

    m = mjx.put_model(mjm)
    d = mjx.put_data(mjm, mjd)

    mjx.euler(m, d)

    np.testing.assert_allclose(d.qvel.numpy()[0], 1 + mjm.opt.timestep)


class ImplicitIntegratorTest(parameterized.TestCase):
  def _load(self, fname: str, disableFlags: int):
    path = epath.resource_path("mujoco.mjx") / "test_data" / fname
    mjm = mujoco.MjModel.from_xml_path(path.as_posix())
    mjm.opt.jacobian = 0
    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    mjm.opt.disableflags = mjm.opt.disableflags | disableFlags
    mjm.actuator_gainprm[:, 2] = np.random.normal(
      scale=10, size=mjm.actuator_gainprm[:, 2].shape
    )

    # change actuators to velocity/damper to cover all codepaths
    mjm.actuator_gaintype[3] = mujoco.mjtGain.mjGAIN_AFFINE
    mjm.actuator_gaintype[6] = mujoco.mjtGain.mjGAIN_AFFINE
    mjm.actuator_biastype[0:3] = mujoco.mjtBias.mjBIAS_AFFINE
    mjm.actuator_biastype[4:6] = mujoco.mjtBias.mjBIAS_AFFINE
    mjm.actuator_biasprm[0:3, 2] = -1
    mjm.actuator_biasprm[4:6, 2] = -1
    mjm.actuator_ctrlrange[3:7] = 10.0
    mjm.actuator_gear[:] = 1.0

    mjd = mujoco.MjData(mjm)

    mjd.qvel = np.random.uniform(low=-0.01, high=0.01, size=mjd.qvel.shape)
    mjd.ctrl = np.random.normal(scale=10, size=mjd.ctrl.shape)
    mjd.act = np.random.normal(scale=10, size=mjd.act.shape)
    mujoco.mj_forward(mjm, mjd)

    mjd.ctrl = np.random.normal(scale=10, size=mjd.ctrl.shape)
    mjd.act = np.random.normal(scale=10, size=mjd.act.shape)
    m = mjx.put_model(mjm)
    d = mjx.put_data(mjm, mjd)
    return mjm, mjd, m, d

  @parameterized.parameters(
    0,
    mjx.DisableBit.PASSIVE.value,
    mjx.DisableBit.ACTUATION.value,
    mjx.DisableBit.PASSIVE.value & mjx.DisableBit.ACTUATION.value,
  )
  def test_implicit(self, disableFlags):
    np.random.seed(0)
    mjm, mjd, m, d = self._load("pendula.xml", disableFlags)

    mjx.implicit(m, d)
    mujoco.mj_implicit(mjm, mjd)

    _assert_eq(d.qpos.numpy()[0], mjd.qpos, "qpos")
    _assert_eq(d.act.numpy()[0], mjd.act, "act")


if __name__ == "__main__":
  wp.init()
  absltest.main()
