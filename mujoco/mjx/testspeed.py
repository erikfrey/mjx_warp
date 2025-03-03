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

"""Run benchmarks on various devices."""

import inspect
from typing import Sequence

from absl import app
from absl import flags
from etils import epath
import warp as wp

import mujoco
from mujoco import mjx

_FUNCTION = flags.DEFINE_enum(
  "function",
  "kinematics",
  [n for n, _ in inspect.getmembers(mjx, inspect.isfunction)],
  "the function to run",
)
_MJCF = flags.DEFINE_string(
  "mjcf", None, "path to model `.xml` or `.mjb`", required=True
)
_BASE_PATH = flags.DEFINE_string(
  "base_path", None, "base path, defaults to mujoco.mjx resource path"
)
_NSTEP = flags.DEFINE_integer("nstep", 1000, "number of steps per rollout")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 4096, "number of parallel rollouts")
_UNROLL = flags.DEFINE_integer("unroll", 1, "loop unroll length")
_SOLVER = flags.DEFINE_enum("solver", "cg", ["cg", "newton"], "constraint solver")
_ITERATIONS = flags.DEFINE_integer("iterations", 1, "number of solver iterations")
_LS_ITERATIONS = flags.DEFINE_integer(
  "ls_iterations", 4, "number of linesearch iterations"
)
_IS_SPARSE = flags.DEFINE_bool(
  "is_sparse", True, "if model should create sparse mass matrices"
)
_NCONMAX = flags.DEFINE_integer(
  "nconmax", -1, "Maximum number of contacts in a batch physics step."
)
_NJMAX = flags.DEFINE_integer(
  "njmax", -1, "Maximum number of constraints in a batch physics step."
)
_OUTPUT = flags.DEFINE_enum(
  "output", "text", ["text", "tsv"], "format to print results"
)
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool(
  "clear_kernel_cache", False, "Clear kernel cache (to calculate full JIT time)"
)


def _main(argv: Sequence[str]):
  """Runs testpeed function."""
  wp.init()

  path = epath.resource_path("mujoco.mjx") / "test_data"
  path = _BASE_PATH.value or path
  f = epath.Path(path) / _MJCF.value
  if f.suffix == ".mjb":
    m = mujoco.MjModel.from_binary_path(f.as_posix())
  else:
    m = mujoco.MjModel.from_xml_path(f.as_posix())

  if _IS_SPARSE.value:
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
  else:
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE

  d = mujoco.MjData(m)
  if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d, 0)
  # populate some constraints
  mujoco.mj_forward(m, d)

  if _CLEAR_KERNEL_CACHE.value:
    wp.clear_kernel_cache()

  print(
    f"Model nbody: {m.nbody} nv: {m.nv} ngeom: {m.ngeom} is_sparse: {_IS_SPARSE.value}"
  )
  print(f"Data ncon: {d.ncon} nefc: {d.nefc}")
  print(f"Rolling out {_NSTEP.value} steps at dt = {m.opt.timestep:.3f}...")
  jit_time, run_time, steps = mjx.benchmark(
    mjx.__dict__[_FUNCTION.value],
    m,
    d,
    _NSTEP.value,
    _BATCH_SIZE.value,
    _UNROLL.value,
    _SOLVER.value,
    _ITERATIONS.value,
    _LS_ITERATIONS.value,
    _NCONMAX.value,
    _NJMAX.value,
  )

  name = argv[0]
  if _OUTPUT.value == "text":
    print(f"""
Summary for {_BATCH_SIZE.value} parallel rollouts

 Total JIT time: {jit_time:.2f} s
 Total simulation time: {run_time:.2f} s
 Total steps per second: {steps / run_time:,.0f}
 Total realtime factor: {steps * m.opt.timestep / run_time:,.2f} x
 Total time per step: {1e9 * run_time / steps:.2f} ns""")
  elif _OUTPUT.value == "tsv":
    name = name.split("/")[-1].replace("testspeed_", "")
    print(f"{name}\tjit: {jit_time:.2f}s\tsteps/second: {steps / run_time:.0f}")


def main():
  app.run(_main)


if __name__ == "__main__":
  main()
