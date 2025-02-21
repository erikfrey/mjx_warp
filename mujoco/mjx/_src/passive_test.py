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

"""Tests for passive force functions."""

from absl.testing import absltest
import numpy as np
import warp as wp

from mujoco import mjx

from . import test_util

# tolerance for difference between MuJoCo and MJX smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class PassiveTest(absltest.TestCase):
  def test_passive(self):
    """Tests MJX passive."""
    _, mjd, m, d = test_util.fixture("pendula.xml")

    for arr in (d.qfrc_spring, d.qfrc_damper, d.qfrc_passive):
      arr.zero_()

    mjx.passive(m, d)

    _assert_eq(d.qfrc_spring.numpy()[0], mjd.qfrc_spring, "qfrc_spring")
    _assert_eq(d.qfrc_damper.numpy()[0], mjd.qfrc_damper, "qfrc_damper")
    _assert_eq(d.qfrc_passive.numpy()[0], mjd.qfrc_passive, "qfrc_passive")


if __name__ == "__main__":
  wp.init()
  absltest.main()
