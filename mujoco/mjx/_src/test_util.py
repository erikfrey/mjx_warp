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

"""Utilities for testing."""

import time
from typing import Callable, Tuple

from etils import epath
import numpy as np
import warp as wp

import mujoco

from . import io
from . import types


def fixture(fname: str, keyframe: int = -1, sparse: bool = True):
  path = epath.resource_path("mujoco.mjx") / "test_data" / fname
  mjm = mujoco.MjModel.from_xml_path(path.as_posix())
  mjd = mujoco.MjData(mjm)
  if keyframe > -1:
    mujoco.mj_resetDataKeyframe(mjm, mjd, keyframe)
  # give the system a little kick to ensure we have non-identity rotations
  mjd.qvel = np.random.uniform(-0.01, 0.01, mjm.nv)
  mujoco.mj_step(mjm, mjd, 3)  # let dynamics get state significantly non-zero
  mujoco.mj_forward(mjm, mjd)
  mjm.opt.jacobian = sparse
  m = io.put_model(mjm)
  d = io.put_data(mjm, mjd)
  return mjm, mjd, m, d


def benchmark(
  fn: Callable[[types.Model, types.Data], None],
  m: mujoco.MjModel,
  nstep: int = 1000,
  batch_size: int = 1024,
  unroll_steps: int = 1,
  solver: str = "cg",
  iterations: int = 1,
  ls_iterations: int = 4,
  nefc_total: int = 0,
) -> Tuple[float, float, int]:
  """Benchmark a model."""

  if solver == "cg":
    m.opt.solver = mujoco.mjtSolver.mjSOL_CG
  elif solver == "newton":
    m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON

  m.opt.iterations = iterations
  m.opt.ls_iterations = ls_iterations

  mx = io.put_model(m)
  dx = io.make_data(m, nworld=batch_size, njmax=nefc_total)
  dx.nefc_total = wp.array([nefc_total], dtype=wp.int32, ndim=1)

  jit_beg = time.perf_counter()
  fn(mx, dx)
  fn(mx, dx) # double warmup to work around issues with compilation during graph capture
  jit_end = time.perf_counter()
  jit_duration = jit_end - jit_beg
  wp.synchronize()

  # capture the whole smooth.kinematic() function as a CUDA graph
  beg, end = None, None
  # beg = wp.Event(enable_timing=True)
  # end = wp.Event(enable_timing=True)
  with wp.ScopedCapture() as capture:
    beg = wp.record_event()
    fn(mx, dx)
    end = wp.record_event()
  graph = capture.graph
  result = wp.get_event_elapsed_time(beg, end, synchronize=True)
  print(result)

  run_beg = time.perf_counter()
  for _ in range(nstep):
    wp.capture_launch(graph)
  wp.synchronize()
  run_end = time.perf_counter()
  run_duration = run_end - run_beg

  return jit_duration, run_duration, batch_size * nstep
