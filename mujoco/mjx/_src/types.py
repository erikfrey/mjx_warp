import warp as wp
import mujoco
from mujoco import mjx

class vec10f(wp.types.vector(length=10, dtype=wp.float32)):
    pass

vec10 = vec10f

@wp.struct
class Model:
  nq: int
  nv: int
  na: int
  nbody: int
  njnt: int
  ngeom: int
  nsite: int
  nmocap: int
  nlevel: int  # warp only
  timestep: float
  qpos0: wp.array(dtype=wp.float32, ndim=1)
  level_beg: wp.array(dtype=wp.int32, ndim=1)  # warp only
  level_beg_cpu: wp.array(dtype=wp.int32, ndim=1)  # warp only
  level_end: wp.array(dtype=wp.int32, ndim=1)    # warp only
  level_end_cpu: wp.array(dtype=wp.int32, ndim=1)  # warp only
  body_bfs: wp.array(dtype=wp.int32, ndim=1)   # warp only
  body_jntadr: wp.array(dtype=wp.int32, ndim=1)
  body_jntnum: wp.array(dtype=wp.int32, ndim=1)
  body_parentid: wp.array(dtype=wp.int32, ndim=1)
  body_mocapid: wp.array(dtype=wp.int32, ndim=1)
  body_pos: wp.array(dtype=wp.vec3, ndim=1)
  body_quat: wp.array(dtype=wp.quat, ndim=1)
  body_ipos: wp.array(dtype=wp.vec3, ndim=1)
  body_iquat: wp.array(dtype=wp.quat, ndim=1)
  body_rootid: wp.array(dtype=wp.int32, ndim=1)
  body_inertia: wp.array(dtype=wp.vec3, ndim=1)
  body_mass: wp.array(dtype=wp.float32, ndim=1)
  jnt_bodyid: wp.array(dtype=wp.int32, ndim=1)
  jnt_type: wp.array(dtype=wp.int32, ndim=1)
  jnt_qposadr: wp.array(dtype=wp.int32, ndim=1)
  jnt_dofadr: wp.array(dtype=wp.int32, ndim=1)
  jnt_axis: wp.array(dtype=wp.vec3, ndim=1)
  jnt_pos: wp.array(dtype=wp.vec3, ndim=1)
  geom_pos: wp.array(dtype=wp.vec3, ndim=1)
  geom_quat: wp.array(dtype=wp.quat, ndim=1)
  site_pos: wp.array(dtype=wp.vec3, ndim=1)
  site_quat: wp.array(dtype=wp.quat, ndim=1)
  dof_bodyid: wp.array(dtype=wp.int32, ndim=1)
  dof_parentid: wp.array(dtype=wp.int32, ndim=1)
  dof_Madr: wp.array(dtype=wp.int32, ndim=1)
  dof_armature: wp.array(dtype=wp.float32, ndim=1)


@wp.struct
class Data:
  nworld: int
  time: float
  qpos: wp.array(dtype=wp.float32, ndim=2)
  mocap_pos: wp.array(dtype=wp.vec3, ndim=2)
  mocap_quat: wp.array(dtype=wp.quat, ndim=2)
  xanchor: wp.array(dtype=wp.vec3, ndim=2)
  xaxis: wp.array(dtype=wp.vec3, ndim=2)
  xmat: wp.array(dtype=wp.mat33, ndim=2)
  xpos: wp.array(dtype=wp.vec3, ndim=2)
  xquat: wp.array(dtype=wp.quat, ndim=2)
  xipos: wp.array(dtype=wp.vec3, ndim=2)
  ximat: wp.array(dtype=wp.mat33, ndim=2)
  subtree_com: wp.array(dtype=wp.vec3, ndim=2)
  geom_xpos: wp.array(dtype=wp.vec3, ndim=2)
  geom_xmat: wp.array(dtype=wp.mat33, ndim=2)
  site_xpos: wp.array(dtype=wp.vec3, ndim=2)
  site_xmat: wp.array(dtype=wp.mat33, ndim=2)
  cinert: wp.array(dtype=vec10, ndim=2)
  cdof: wp.array(dtype=wp.spatial_vector, ndim=2)
  crb: wp.array(dtype=vec10, ndim=2)
  qM: wp.array(dtype=wp.float32, ndim=2)
  qacc: wp.array(dtype=wp.float32, ndim=2)
  qvel: wp.array(dtype=wp.float32, ndim=2)
  act_dot: wp.array(dtype=wp.float32, ndim=2)
  act: wp.array(dtype=wp.float32, ndim=2)
