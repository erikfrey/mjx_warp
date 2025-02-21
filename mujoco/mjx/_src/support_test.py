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

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
import numpy as np
import warp as wp

from . import test_util
from .support import xfrc_accumulate

class SmoothTest(parameterized.TestCase):

 def test_xfrc_accumulated(self):
    """Tests that xfrc_accumulate ouput matches mj_xfrcAccumulate."""
    np.random.seed(0)
    mjm, mjd, m, d = test_util.fixture('pendula.xml')
    xfrc = np.random.randn(*d.xfrc_applied.numpy().shape)
    d.xfrc_applied = wp.from_numpy(xfrc, dtype=wp.spatial_vector)
    qfrc = xfrc_accumulate(m, d)

    qfrc_expected = np.zeros(m.nv)
    xfrc =xfrc[0]
    mjd.xfrc_applied[:] = xfrc
    for i in range(1, m.nbody):
        mujoco.mj_applyFT(
            mjm,
            mjd,
            mjd.xfrc_applied[i, :3],
            mjd.xfrc_applied[i, 3:],
            mjd.xipos[i],
            i,
            qfrc_expected,
        )
    np.testing.assert_almost_equal(qfrc.numpy()[0], qfrc_expected, 6)

if __name__ == '__main__':
  wp.init()
  absltest.main()
