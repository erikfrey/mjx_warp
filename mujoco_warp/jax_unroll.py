import jax
from jax import numpy as jp
from etils import epath
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from warp.jax_experimental.ffi import jax_callable


def main():
  path = epath.resource_path("mujoco_warp") / "test_data" / 'humanoid/humanoid.xml'
  mjm = mujoco.MjModel.from_xml_path(path.as_posix())
  mjd = mujoco.MjData(mjm)
  # give the system a little kick to ensure we have non-identity rotations
  mjd.qvel = np.random.uniform(-0.01, 0.01, mjm.nv)
  mujoco.mj_step(mjm, mjd, 3)  # let dynamics get state significantly non-zero
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=8192, nconmax=131012, njmax=131012 * 4)


  def warp_step(qpos_in: wp.array(dtype=wp.float32, ndim=2),
                qvel_in: wp.array(dtype=wp.float32, ndim=2),
                qpos_out: wp.array(dtype=wp.float32, ndim=2),
                qvel_out: wp.array(dtype=wp.float32, ndim=2)):
    wp.copy(d.qpos, qpos_in)
    wp.copy(d.qvel, qvel_in)
    mjw.step(m, d)
    wp.copy(qpos_out, d.qpos)
    wp.copy(qvel_out, d.qvel)

  warp_step_fn = jax_callable(warp_step, num_outputs=2, output_dims={'qpos_out': (8192, 28), 'qvel_out': (8192, 27)})

  jax_qpos = jp.tile(jp.array(m.qpos0), (8192, 1))
  jax_qvel = jp.zeros((8192, m.nv))

  # raise me for slowdown:
  unroll_length = 1

  def unroll(qpos, qvel):

    def step(carry, _):
      qpos, qvel = carry
      qpos, qvel = warp_step_fn(qpos, qvel)
      return (qpos, qvel), None

    (qpos, qvel), _ = jax.lax.scan(step, (qpos, qvel), length=unroll_length)

    return qpos, qvel

  jax_unroll_fn = jax.jit(unroll)

  next_jax_qpos, next_jax_qvel = jax_unroll_fn(jax_qpos, jax_qvel)


if __name__ == "__main__":
  main()
