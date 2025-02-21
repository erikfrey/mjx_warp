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

"""Tests for constraint functions."""

from absl.testing import absltest
from absl.testing import parameterized
from . import test_util
import mujoco
from mujoco import mjx
import numpy as np

# tolerance for difference between MuJoCo and MJX constraint calculations,
# mostly due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class ConstraintTest(parameterized.TestCase):
  def setUp(self):
    super().setUp()
    np.random.seed(42)

  @parameterized.parameters(
    {
      "cone": mujoco.mjtCone.mjCONE_PYRAMIDAL,
      "rand_eq_active": False,
      "fname": "humanoid/humanoid.xml",
    },
  )
  def test_constraints(self, cone, rand_eq_active, fname: str):
    """Test constraints."""
    m = test_util.load_test_file("constraints.xml")
    m.opt.cone = cone
    d = mujoco.MjData(m)

    # sample a mix of active/inactive constraints at different timesteps
    for key in range(3):
      mujoco.mj_resetDataKeyframe(m, d, key)
      if rand_eq_active:
        d.eq_active[:] = np.random.randint(0, 2, size=m.neq)

      mujoco.mj_forward(m, d)
      mx = mjx.put_model(m)
      dx = mjx.put_data(m, d)
      dx = mjx.make_constraint(mx, dx)

      _assert_eq(d.efc_J, np.reshape(dx.efc_J.numpy(), shape=(d.nefc * m.nv)), "efc_J")
      _assert_eq(d.efc_D, np.reshape(dx.efc_D.numpy(), shape=(d.nefc)), "efc_D")
      _assert_eq(
        d.efc_aref, np.reshape(dx.efc_aref.numpy(), shape=(d.nefc)), "efc_aref"
      )
      _assert_eq(d.efc_pos, np.reshape(dx.efc_pos.numpy(), shape=(d.nefc)), "efc_pos")
      _assert_eq(
        d.efc_margin, np.reshape(dx.efc_margin.numpy(), shape=(d.nefc)), "efc_margin"
      )
      _assert_eq(
        d.efc_frictionloss,
        np.reshape(dx.efc_frictionloss.numpy(), shape=(d.nefc)),
        "efc_frictionloss",
      )


if __name__ == "__main__":
  absltest.main()
