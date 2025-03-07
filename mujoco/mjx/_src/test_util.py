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
from . import warp_util


def fixture(fname: str, keyframe: int = -1, sparse: bool = True):
  path = epath.resource_path("mujoco.mjx") / "test_data" / fname
  mjm = mujoco.MjModel.from_xml_path(path.as_posix())
  mjm.opt.jacobian = sparse
  mjd = mujoco.MjData(mjm)
  if keyframe > -1:
    mujoco.mj_resetDataKeyframe(mjm, mjd, keyframe)
  # give the system a little kick to ensure we have non-identity rotations
  mjd.qvel = np.random.uniform(-0.01, 0.01, mjm.nv)
  mujoco.mj_step(mjm, mjd, 3)  # let dynamics get state significantly non-zero
  mujoco.mj_forward(mjm, mjd)
  m = io.put_model(mjm)
  d = io.put_data(mjm, mjd)
  return mjm, mjd, m, d


def _sum(stack1, stack2):
  ret = {}
  for k in stack1:
    times1, sub_stack1 = stack1[k]
    times2, sub_stack2 = stack2[k]
    times = [t1 + t2 for t1, t2 in zip(times1, times2)]
    ret[k] = (times, _sum(sub_stack1, sub_stack2))
  return ret


def benchmark(
  fn: Callable[[types.Model, types.Data], None],
  mjm: mujoco.MjModel,
  mjd: mujoco.MjData,
  nstep: int = 1000,
  batch_size: int = 1024,
  solver: str = "cg",
  iterations: int = 1,
  ls_iterations: int = 4,
  nconmax: int = -1,
  njmax: int = -1,
  event_trace: bool = False,
) -> Tuple[float, float, dict, int]:
  """Benchmark a model."""

  if solver == "cg":
    mjm.opt.solver = mujoco.mjtSolver.mjSOL_CG
  elif solver == "newton":
    mjm.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON

  mjm.opt.iterations = iterations
  mjm.opt.ls_iterations = ls_iterations

  m = io.put_model(mjm)
  d = io.put_data(mjm, mjd, nworld=batch_size, nconmax=nconmax, njmax=njmax)

  jit_beg = time.perf_counter()

  fn(m, d)
  # double warmup to work around issues with compilation during graph capture:
  fn(m, d)

  jit_end = time.perf_counter()
  jit_duration = jit_end - jit_beg
  wp.synchronize()
  trace = {}

  with warp_util.EventTracer(enabled=event_trace) as tracer:
    # capture the whole smooth.kinematic() function as a CUDA graph
    with wp.ScopedCapture() as capture:
      fn(m, d)
    graph = capture.graph

    run_beg = time.perf_counter()
    for _ in range(nstep):
      wp.capture_launch(graph)
      if trace:
        trace = _sum(trace, tracer.trace())
      else:
        trace = tracer.trace()
    wp.synchronize()
    run_end = time.perf_counter()
    run_duration = run_end - run_beg

  return jit_duration, run_duration, trace, batch_size * nstep
