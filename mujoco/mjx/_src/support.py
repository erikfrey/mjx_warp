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


import mujoco
import warp as wp
from .types import Model
from .types import Data

def is_sparse(m: mujoco.MjModel):
  if m.opt.jacobian == mujoco.mjtJacobian.mjJAC_AUTO:
    return m.nv >= 60
  return m.opt.jacobian == mujoco.mjtJacobian.mjJAC_SPARSE

@wp.kernel
def process_level(
    body_tree: wp.array(dtype=int),
    body_parentid: wp.array(dtype=int),
    dof_bodyid: wp.array(dtype=int),
    matrix: wp.array2d(dtype=wp.bool),
    beg: int,
):
    tid_x, tid_y = wp.tid()
    j = beg + tid_x
    dofid = tid_y 

    el = body_tree[j]
    parent_id = body_parentid[el]
    parent_val = matrix[parent_id, dofid]
    matrix[el, dofid] = parent_val or (dof_bodyid[dofid] == el)

@wp.kernel
def compute_jacobians(
    d:Data,
    m:Model,
    mask: wp.array2d(dtype=wp.bool),
    jacp: wp.array3d(dtype=wp.vec3),
    jacr: wp.array3d(dtype=wp.vec3)
):
    worldid, bodyid, dofid = wp.tid()
   
    offset = d.xipos[worldid, bodyid] - d.subtree_com[worldid,  m.body_rootid[bodyid]]
    cdof_vec = d.cdof[worldid, dofid]  
    angular = wp.vec3(cdof_vec[0], cdof_vec[1], cdof_vec[2])
    linear = wp.vec3(cdof_vec[3], cdof_vec[4], cdof_vec[5])
    jacp[worldid, bodyid, dofid] = (linear + wp.cross(angular, offset)) * wp.float(mask[bodyid, dofid])
    jacr[worldid, bodyid, dofid] = angular *  wp.float(mask[bodyid, dofid])

@wp.kernel
def compute_qfrc(
    d:Data,
    jacp:  wp.array3d(dtype=wp.vec3),   
    jacr: wp.array3d(dtype=wp.vec3),  
    qfrc_total: wp.array2d(dtype=float)
):
    worldid, bodyid, dofid = wp.tid()
    jacp_vec = jacp[worldid, bodyid, dofid]
    jacr_vec = jacr[worldid, bodyid, dofid]
    xfrc_applied_vec = d.xfrc_applied[worldid, bodyid]
    
    jacp_force = (
        jacp_vec[0] * xfrc_applied_vec[0] +
        jacp_vec[1] * xfrc_applied_vec[1] +
        jacp_vec[2] * xfrc_applied_vec[2] 
    )
    
    jacr_torque = (
        jacr_vec[0] * xfrc_applied_vec[3] +
        jacr_vec[1] * xfrc_applied_vec[4] +
        jacr_vec[2] * xfrc_applied_vec[5]
    )
    
    wp.atomic_add(qfrc_total[worldid], dofid, jacp_force + jacr_torque)


def xfrc_accumulate(m: Model, d: Data):

    body_tree_np = wp.from_numpy(m.body_tree.numpy().astype(int))
    body_treeadr_np = m.body_treeadr.numpy()
    
    # compute mask matrix
    mask = wp.zeros((m.nbody, m.nv), dtype=wp.bool)
    for i in range(len(body_treeadr_np)):
        beg = body_treeadr_np[i]
        end = body_treeadr_np[i+1] if i < len(body_treeadr_np)-1 else len(body_tree_np)
        
        if end > beg:
            wp.launch(
                kernel=process_level,
                dim=[(end - beg), m.nv],
                inputs=[
                    m.body_tree,
                    m.body_parentid,
                    m.dof_bodyid,
                    mask,
                    beg
                ]
            )


    # compute jac using precomputed mask
    jacp = wp.zeros(shape=(d.nworld, m.nbody, m.nv), dtype=wp.vec3)
    jacr = wp.zeros_like(jacp)
    wp.launch(
        kernel=compute_jacobians,
        dim=(d.nworld,  m.nbody, m.nv),
        inputs=[d,
            m,
            mask, 
            jacp, 
            jacr]
    )

    # multiply forces with precomputed jcap and jcar to get qfrc
    qfrc_total = wp.zeros((d.nworld, m.nv), dtype=float)
    wp.launch(
        kernel=compute_qfrc,
        dim=(d.nworld, m.nbody, m.nv),
        inputs=[d, jacp, jacr, qfrc_total]
    )

    return qfrc_total