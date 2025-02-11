"""Tests for forward functions."""

from absl.testing import absltest
from etils import epath
import mujoco
from mujoco import mjx
import numpy as np


# tolerance for difference between MuJoCo and MJX forward calculations - mostly
# due to float precision
_TOLERANCE = 1e-5

def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f'mismatch: {name}'
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)

class ForwardTest(absltest.TestCase):
  
  def test_euler(self):
    path = epath.resource_path('mujoco.mjx') / 'test_data/constraints.xml'
    mjm = mujoco.MjModel.from_xml_path(path.as_posix())
    # disable euler damp
    #mjm.opt.disableflags = mjm.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_EULERDAMP

    mjd = mujoco.MjData(mjm)
    # apply some control and xfrc input
    mjd.ctrl = np.array([-18, 0.59, 0.47])
    mjd.xfrc_applied[0, 2] = 0.1  # torque
    mjd.xfrc_applied[1, 4] = 0.3  # linear force
    mujoco.mj_step(mjm, mjd, 20)  # get some dynamics going
    mujoco.mj_forward(mjm, mjd)

    m = mjx.put_model(mjm)
    d = mjx.put_data(mjm, mjd)

    # euler
    mjx.euler(m, d)
    mujoco.mj_Euler(mjm, mjd)
    _assert_eq(d.act.numpy()[0], mjd.act, 'act')
    _assert_eq(d.qpos.numpy()[0], mjd.qpos, 'qpos')
    _assert_eq(d.time, mjd.time, 'time')
