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

from absl.testing import absltest

import mujoco
from mujoco import mjx


class ConvexTest(absltest.TestCase):
  """Tests the convex contact functions."""

  _BOX_PLANE = """
    <mujoco>
      <worldbody>
        <geom size="40 40 40" type="plane"/>
        <body pos="0 0 0.7" euler="45 0 0">
          <freejoint/>
          <geom size="0.5 0.5 0.5" type="box"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_box_plane(self):
    """Tests box collision with a plane."""
    m = mujoco.MjModel.from_xml_string(self._BOX_PLANE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    mjx.collision(mx, dx)

  _CYLINDER_CYLINDER = """
    <mujoco>
      <worldbody>
        <body pos="0.5 1.0 0" euler="0 0 0">
          <freejoint/>
          <geom size="0.5 0.5" type="cylinder"/>
        </body>
        <body pos="0.5 1.0 0" euler="0 90 0">
          <freejoint/>
          <geom size="0.5 1.0" type="cylinder"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_cylinder_cylinder(self):
    """Tests cylinder collision with a cylinder."""
    m = mujoco.MjModel.from_xml_string(self._CYLINDER_CYLINDER)
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d, nconmax=10)

    mjx.collision(mx, dx)
    print(dx.contact)
    assert 0 == 1
