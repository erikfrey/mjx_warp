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
"""Tests the collision driver."""

import mujoco
from mujoco import mjx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized


class PrimitiveTest(parameterized.TestCase):
  """Tests the primtive contact functions."""

  _MJCFS = {
    "plane_sphere": """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane"/>
            <body pos="0 0 0.2" euler="45 0 0">
              <freejoint/>
              <geom size="0.5" type="sphere"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    "sphere_sphere": """
        <mujoco>
          <worldbody>
            <body>
              <joint type="free"/>
              <geom pos="0 0 0" size="0.2" type="sphere"/>
            </body>
            <body >
              <joint type="free"/>
              <geom pos="0 0.3 0" size="0.11" type="sphere"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    "capsule_capsule": """
        <mujoco model="two_capsules">
          <worldbody>
            <body>
              <joint type="free"/>
              <geom fromto="0.62235904  0.58846647 0.651046 1.5330081 0.33564585 0.977849"
               size="0.05" type="capsule"/>
            </body>
            <body>
              <joint type="free"/>
              <geom fromto="0.5505271 0.60345304 0.476661 1.3900293 0.30709633 0.932082"
               size="0.05" type="capsule"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    "plane_capsule": """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane"/>
            <body pos="0 0 0.0" euler="30 30 0">
              <freejoint/>
              <geom size="0.05 0.05" type="capsule"/>
            </body>
          </worldbody>
        </mujoco>
        """,
  }

  @parameterized.parameters(
    "plane_sphere",
    "sphere_sphere",
    "plane_capsule",
    "capsule_capsule",
  )
  def test_contact(self, name):
    """Tests contact calculation with different collision functions."""
    m = mujoco.MjModel.from_xml_string(self._MJCFS[name])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)
    mjx.collision(mx, dx)
    mujoco.mj_collision(m, d)
    self.assertEqual(d.ncon, dx.ncon.numpy()[0])
    for i in range(d.ncon):
      actual_dist = dx.contact.dist.numpy()[i]
      actual_pos = dx.contact.pos.numpy()[i, :]
      actual_frame = dx.contact.frame.numpy()[i].flatten()
      np.testing.assert_array_almost_equal(actual_dist, d.contact.dist[i], 4)
      np.testing.assert_array_almost_equal(actual_pos, d.contact.pos[i], 4)
      np.testing.assert_array_almost_equal(actual_frame, d.contact.frame[i], 4)


if __name__ == "__main__":
  absltest.main()
