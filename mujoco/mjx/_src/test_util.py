# Copyright 2023 DeepMind Technologies Limited
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

import warp as wp

import mujoco

# pylint: disable=g-importing-member
from . import io
from . import smooth
from .types import Data
from .types import Model

# pylint: enable=g-importing-member


def benchmark(
    fn: Callable[[Model, Data], None],
    m: mujoco.MjModel,
    nstep: int = 1000,
    batch_size: int = 1024,
    unroll_steps: int = 1,
    solver: str = "newton",
    iterations: int = 1,
    ls_iterations: int = 4,
) -> Tuple[float, float, int]:
  """Benchmark a model."""

  mx = io.put_model(m)
  dx = io.make_data(m, nworld=batch_size)

  wp.clear_kernel_cache()
  jit_beg = time.perf_counter()
  fn(mx, dx)
  jit_end = time.perf_counter()
  jit_duration = jit_end - jit_beg
  wp.synchronize()

  # capture the whole smooth.kinematic() function as a CUDA graph
  with wp.ScopedCapture() as capture:
    fn(mx, dx)
  graph = capture.graph

  run_beg = time.perf_counter()
  for _ in range(nstep):
    wp.capture_launch(graph)
  wp.synchronize()
  run_end = time.perf_counter()
  run_duration = run_end - run_beg

  return jit_duration, run_duration, batch_size * nstep
