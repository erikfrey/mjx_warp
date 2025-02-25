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

"""Tests for support functions."""

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
from mujoco import mjx
import numpy as np
import warp as wp

wp.config.verify_cuda = True

from . import test_util

# tolerance for difference between MuJoCo and mjWarp support calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class SupportTest(parameterized.TestCase):
  @parameterized.parameters(True, False)
  def test_mul_m(self, sparse):
    """Tests mul_m."""
    mjm, mjd, m, d = test_util.fixture("pendula.xml", sparse=sparse)

    mj_res = np.zeros(mjm.nv)
    mj_vec = np.random.uniform(low=-1.0, high=1.0, size=mjm.nv)
    mujoco.mj_mulM(mjm, mjd, mj_res, mj_vec)

    res = wp.zeros((1, mjm.nv), dtype=wp.float32)
    vec = wp.from_numpy(np.expand_dims(mj_vec, axis=0), dtype=wp.float32)
    mjx.mul_m(m, d, res, vec)

    _assert_eq(res.numpy()[0], mj_res, f"mul_m ({'sparse' if sparse else 'dense'})")


if __name__ == "__main__":
  wp.init()
  absltest.main()
