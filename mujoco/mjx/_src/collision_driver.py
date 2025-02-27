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

import warp as wp

from .types import Model
from .types import Data
from .types import Contact
from .types import BoxType
from .types import GeomType
from .types import MJ_MINVAL
from .types import MJ_NREF
from .collision_functions import plane_sphere
from .collision_functions import plane_capsule
from .collision_functions import plane_convex
from .collision_functions import plane_ellipsoid
from .collision_functions import plane_cylinder
from .collision_functions import plane_convex
from .collision_functions import hfield_sphere
from .collision_functions import hfield_capsule
from .collision_functions import hfield_convex
from .collision_functions import sphere_sphere
from .collision_functions import sphere_capsule
from .collision_functions import sphere_cylinder
from .collision_functions import sphere_ellipsoid
from .collision_functions import sphere_convex
from .collision_functions import capsule_capsule
from .collision_functions import capsule_convex
from .collision_functions import capsule_ellipsoid
from .collision_functions import capsule_cylinder
from .collision_functions import capsule_convex
from .collision_functions import ellipsoid_ellipsoid
from .collision_functions import ellipsoid_cylinder
from .collision_functions import cylinder_cylinder
from .collision_functions import box_box
from .collision_functions import convex_convex

# TODO: combine the two lists, it matters that the order is the same!
_COLLISION_BUCKET_KEY = {
    (GeomType.PLANE, GeomType.SPHERE):        0, #plane_sphere,
    (GeomType.PLANE, GeomType.CAPSULE):       1, #plane_capsule,
    (GeomType.PLANE, GeomType.BOX):           2, #plane_convex,
    (GeomType.PLANE, GeomType.ELLIPSOID):     3, #plane_ellipsoid,
    (GeomType.PLANE, GeomType.CYLINDER):      4, #plane_cylinder,
    (GeomType.PLANE, GeomType.MESH):          5, #plane_convex,
    (GeomType.HFIELD, GeomType.SPHERE):       6, #hfield_sphere,
    (GeomType.HFIELD, GeomType.CAPSULE):      7, #hfield_capsule,
    (GeomType.HFIELD, GeomType.BOX):          8, #hfield_convex,
    (GeomType.HFIELD, GeomType.MESH):         9, #hfield_convex,
    (GeomType.SPHERE, GeomType.SPHERE):       10, #sphere_sphere,
    (GeomType.SPHERE, GeomType.CAPSULE):      11, #sphere_capsule,
    (GeomType.SPHERE, GeomType.CYLINDER):     12, #sphere_cylinder,
    (GeomType.SPHERE, GeomType.ELLIPSOID):    13, #sphere_ellipsoid,
    (GeomType.SPHERE, GeomType.BOX):          14, #sphere_convex,
    (GeomType.SPHERE, GeomType.MESH):         15, #sphere_convex,
    (GeomType.CAPSULE, GeomType.CAPSULE):     16, #capsule_capsule,
    (GeomType.CAPSULE, GeomType.BOX):         17, #capsule_convex,
    (GeomType.CAPSULE, GeomType.ELLIPSOID):   18, #capsule_ellipsoid,
    (GeomType.CAPSULE, GeomType.CYLINDER):    19, #capsule_cylinder,
    (GeomType.CAPSULE, GeomType.MESH):        20, #capsule_convex,
    (GeomType.ELLIPSOID, GeomType.ELLIPSOID): 21, #ellipsoid_ellipsoid,
    (GeomType.ELLIPSOID, GeomType.CYLINDER):  22, #ellipsoid_cylinder,
    (GeomType.CYLINDER, GeomType.CYLINDER):   23, #cylinder_cylinder,
    (GeomType.BOX, GeomType.BOX):             24, #box_box,
    (GeomType.BOX, GeomType.MESH):            25, #convex_convex,
    (GeomType.MESH, GeomType.MESH):           26, #convex_convex,
}

_COLLISION_FUNCS = {
  plane_sphere,
  plane_capsule,
  plane_convex,
  plane_ellipsoid,
  plane_cylinder,
  plane_convex,
  hfield_sphere,
  hfield_capsule,
  hfield_convex,
  hfield_convex,
  sphere_sphere,
  sphere_capsule,
  sphere_cylinder,
  sphere_ellipsoid,
  sphere_convex,
  sphere_convex,
  capsule_capsule,
  capsule_convex,
  capsule_ellipsoid,
  capsule_cylinder,
  capsule_convex,
  ellipsoid_ellipsoid,
  ellipsoid_cylinder,
  cylinder_cylinder,
  box_box,
  convex_convex,
  convex_convex,
}

#####################################################################################
# BROADPHASE
#####################################################################################
# TODO: Verify that this is corect
@wp.func
def transform_aabb(
  aabb: wp.types.matrix(shape=(2, 3), dtype=wp.float32),
  pos: wp.vec3,
  rot: wp.mat33,
) -> wp.types.matrix(shape=(2, 3), dtype=wp.float32):
  # Extract center and extents from AABB
  center = aabb[0]
  extents = aabb[1]

  absRot = rot
  absRot[0, 0] = wp.abs(rot[0, 0])
  absRot[0, 1] = wp.abs(rot[0, 1])
  absRot[0, 2] = wp.abs(rot[0, 2])
  absRot[1, 0] = wp.abs(rot[1, 0])
  absRot[1, 1] = wp.abs(rot[1, 1])
  absRot[1, 2] = wp.abs(rot[1, 2])
  absRot[2, 0] = wp.abs(rot[2, 0])
  absRot[2, 1] = wp.abs(rot[2, 1])
  absRot[2, 2] = wp.abs(rot[2, 2])
  world_extents = extents * absRot

  # Transform center
  new_center = rot @ center + pos

  # Return new AABB as matrix with center and full size
  result = BoxType()
  result[0] = wp.vec3(new_center.x, new_center.y, new_center.z)
  result[1] = wp.vec3(world_extents.x, world_extents.y, world_extents.z)
  return result


@wp.func
def overlap(
  a: wp.types.matrix(shape=(2, 3), dtype=wp.float32),
  b: wp.types.matrix(shape=(2, 3), dtype=wp.float32),
) -> bool:
  # Extract centers and sizes
  a_center = a[0]
  a_size = a[1]
  b_center = b[0]
  b_size = b[1]

  # Calculate min/max from center and size
  a_min = a_center - 0.5 * a_size
  a_max = a_center + 0.5 * a_size
  b_min = b_center - 0.5 * b_size
  b_max = b_center + 0.5 * b_size

  return not (
    a_min.x > b_max.x
    or b_min.x > a_max.x
    or a_min.y > b_max.y
    or b_min.y > a_max.y
    or a_min.z > b_max.z
    or b_min.z > a_max.z
  )

@wp.kernel
def broad_phase_project_boxes_onto_sweep_direction_kernel(
  boxes: wp.array(dtype=wp.types.matrix(shape=(2, 3), dtype=wp.float32), ndim=1),
  box_translations: wp.array(dtype=wp.vec3, ndim=2),
  box_rotations: wp.array(dtype=wp.mat33, ndim=2),
  data_start: wp.array(dtype=wp.float32, ndim=2),
  data_end: wp.array(dtype=wp.float32, ndim=2),
  data_indexer: wp.array(dtype=wp.int32, ndim=2),
  direction: wp.vec3,
  abs_dir: wp.vec3,
  result_count: wp.array(dtype=wp.int32, ndim=1),
):
  worldId, i = wp.tid()

  box = boxes[i]  # box is a vector6
  box = transform_aabb(box, box_translations[worldId, i], box_rotations[worldId, i])
  box_center = box[0]
  box_size = box[1]
  center = wp.dot(direction, box_center)
  d = wp.dot(box_size, abs_dir)
  f = center - d

  # Store results in the data arrays
  data_start[worldId, i] = f
  data_end[worldId, i] = center + d
  data_indexer[worldId, i] = i

  if i == 0:
    result_count[worldId] = 0  # Initialize result count to 0


@wp.kernel
def reorder_bounding_boxes_kernel(
  boxes: wp.array(dtype=wp.types.matrix(shape=(2, 3), dtype=wp.float32), ndim=1),
  box_translations: wp.array(dtype=wp.vec3, ndim=2),
  box_rotations: wp.array(dtype=wp.mat33, ndim=2),
  boxes_sorted: wp.array(dtype=wp.types.matrix(shape=(2, 3), dtype=wp.float32), ndim=2),
  data_indexer: wp.array(dtype=wp.int32, ndim=2),
):
  worldId, i = wp.tid()

  # Get the index from the data indexer
  mapped = data_indexer[worldId, i]

  # Get the box from the original boxes array
  box = boxes[mapped]
  box = transform_aabb(
    box, box_translations[worldId, mapped], box_rotations[worldId, mapped]
  )

  # Reorder the box into the sorted array
  boxes_sorted[worldId, i] = box

@wp.func
def find_first_greater_than(
  worldId: int,
  starts: wp.array(dtype=wp.float32, ndim=2),
  value: wp.float32,
  low: int,
  high: int,
) -> int:
  while low < high:
    mid = (low + high) >> 1
    if starts[worldId, mid] > value:
      high = mid
    else:
      low = mid + 1
  return low

@wp.kernel
def broad_phase_sweep_and_prune_prepare_kernel(
  num_boxes_per_world: int,
  data_start: wp.array(dtype=wp.float32, ndim=2),
  data_end: wp.array(dtype=wp.float32, ndim=2),
  indexer: wp.array(dtype=wp.int32, ndim=2),
  cumulative_sum: wp.array(dtype=wp.int32, ndim=2),
):
  worldId, i = wp.tid()  # Get the thread ID

  # Get the index of the current bounding box
  idx1 = indexer[worldId, i]

  end = data_end[worldId, idx1]
  limit = find_first_greater_than(worldId, data_start, end, i + 1, num_boxes_per_world)
  limit = wp.min(num_boxes_per_world - 1, limit)

  # Calculate the range of boxes for the sweep and prune process
  count = limit - i

  # Store the cumulative sum for the current box
  cumulative_sum[worldId, i] = count

@wp.func
def find_right_most_index_int(
  starts: wp.array(dtype=wp.int32, ndim=1), value: wp.int32, low: int, high: int
) -> int:
  while low < high:
    mid = (low + high) >> 1
    if starts[mid] > value:
      high = mid
    else:
      low = mid + 1
  return high

@wp.func
def find_indices(
  id: int, cumulative_sum: wp.array(dtype=wp.int32, ndim=1), length: int
) -> wp.vec2i:
  # Perform binary search to find the right most index
  i = find_right_most_index_int(cumulative_sum, id, 0, length)

  # Get the baseId, and compute the offset and j
  if i > 0:
    base_id = cumulative_sum[i - 1]
  else:
    base_id = 0
  offset = id - base_id
  j = i + offset + 1

  return wp.vec2i(i, j)

@wp.kernel
def broad_phase_sweep_and_prune_kernel(
  num_threads: int,
  length: int,
  num_boxes_per_world: int,
  max_num_overlaps_per_world: int,
  cumulative_sum: wp.array(dtype=wp.int32, ndim=1),
  data_indexer: wp.array(dtype=wp.int32, ndim=2),
  data_result: wp.array(dtype=wp.vec2i, ndim=2),
  result_count: wp.array(dtype=wp.int32, ndim=1),
  boxes_sorted: wp.array(dtype=wp.types.matrix(shape=(2, 3), dtype=wp.float32), ndim=2),
):
  threadId = wp.tid()  # Get thread ID
  if length > 0:
    total_num_work_packages = cumulative_sum[length - 1]
  else:
    total_num_work_packages = 0

  while threadId < total_num_work_packages:
    # Get indices for current and next box pair
    ij = find_indices(threadId, cumulative_sum, length)
    i = ij.x
    j = ij.y

    worldId = i // num_boxes_per_world
    i = i % num_boxes_per_world

    # world_id_j = j // num_boxes_per_world
    j = j % num_boxes_per_world

    # assert worldId == world_id_j, "Only boxes in the same world can be compared"
    # TODO: Remove print if debugging is done
    # if worldId != world_id_j:
    #     print("Only boxes in the same world can be compared")

    idx1 = data_indexer[worldId, i]

    box1 = boxes_sorted[worldId, i]

    idx2 = data_indexer[worldId, j]

    # Check if the boxes overlap
    if idx1 != idx2 and overlap(box1, boxes_sorted[worldId, j]):
      pair = wp.vec2i(wp.min(idx1, idx2), wp.max(idx1, idx2))

      id = wp.atomic_add(result_count, worldId, 1)

      if id < max_num_overlaps_per_world:
        data_result[worldId, id] = pair

    threadId += num_threads



def broad_phase(m: Model, d: Data):
  """Broad phase collision detection."""

  # Directional vectors for sweep
  # TODO: Improve picking of direction
  direction = wp.vec3(0.5935, 0.7790, 0.1235)
  direction = wp.normalize(direction)
  abs_dir = wp.vec3(abs(direction.x), abs(direction.y), abs(direction.z))

  wp.launch(
    kernel=broad_phase_project_boxes_onto_sweep_direction_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[
      d.geom_aabb,
      d.geom_xpos,
      d.geom_xmat,
      d.data_start,
      d.data_end,
      d.data_indexer,
      direction,
      abs_dir,
      d.result_count,
    ],
  )

  segmented_sort_available = hasattr(wp.utils, "segmented_sort_pairs")

  if segmented_sort_available:
    # print("Using segmented sort")
    wp.utils.segmented_sort_pairs(
      d.data_start,
      d.data_indexer,
      m.ngeom * d.nworld,
      d.segment_indices,
      d.nworld,
    )
  else:
    # Sort each world's segment separately
    for world_id in range(d.nworld):
      start_idx = world_id * m.ngeom

      # Create temporary arrays for sorting
      temp_data_start = wp.zeros(
        m.ngeom * 2,
        dtype=d.data_start.dtype,
      )
      temp_data_indexer = wp.zeros(
        m.ngeom * 2,
        dtype=d.data_indexer.dtype,
      )

      # Copy data to temporary arrays
      wp.copy(
        temp_data_start,
        d.data_start,
        0,
        start_idx,
        m.ngeom,
      )
      wp.copy(
        temp_data_indexer,
        d.data_indexer,
        0,
        start_idx,
        m.ngeom,
      )

      # Sort the temporary arrays
      wp.utils.radix_sort_pairs(temp_data_start, temp_data_indexer, m.ngeom)

      # Copy sorted data back
      wp.copy(
        d.data_start,
        temp_data_start,
        start_idx,
        0,
        m.ngeom,
      )
      wp.copy(
        d.data_indexer,
        temp_data_indexer,
        start_idx,
        0,
        m.ngeom,
      )

  wp.launch(
    kernel=reorder_bounding_boxes_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[d.geom_aabb, d.geom_xpos, d.geom_xmat, d.boxes_sorted, d.data_indexer],
  )

  wp.launch(
    kernel=broad_phase_sweep_and_prune_prepare_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[
      m.ngeom,
      d.data_start,
      d.data_end,
      d.data_indexer,
      d.ranges,
    ],
  )

  # The scan (scan = cumulative sum, either inclusive or exclusive depending on the last argument) is used for load balancing among the threads
  wp.utils.array_scan(d.ranges.reshape(-1), d.cumulative_sum, True)

  # Estimate how many overlap checks need to be done - assumes each box has to be compared to 5 other boxes (and batched over all worlds)
  num_sweep_threads = 5 * d.nworld * m.ngeom
  wp.launch(
    kernel=broad_phase_sweep_and_prune_kernel,
    dim=num_sweep_threads,
    inputs=[
      num_sweep_threads,
      d.nworld * m.ngeom,
      m.ngeom,
      d.max_num_overlaps_per_world,
      d.cumulative_sum,
      d.data_indexer,
      d.broadphase_pairs,
      d.result_count,
      d.boxes_sorted,
    ],
  )

##############################################################################################
# FILTERING
##############################################################################################

def filtering(m: Model, d: Data) -> Data:
  # takes overlap pairs and filters them, assigns pairs to type-pairs

  # this is the place where we do the stuff done in the MJX geom_groups/contact_groups functions
  pass

##############################################################################################
# NARROW PHASE PREPROCESSING
##############################################################################################

def overlaps_to_type_buckets(m: Model, d: Data) -> Data:
  # pair type key is _COLLISION_BUCKET_KEY

  # takes the final list of pairs, puts them into buckets per type-pair.
  # maintain counts per type-pair.
  # make sure the worldid is taken along for the ride postprocessing.

  # is this the place where we also fill in per-pair options from the model?
  pass

##############################################################################################
# NARROW PHASE
#############################################################################################

def narrow_phase(m: Model, d: Data) -> Data:

  # we need to figure out how to keep the overhead of this small - not launching anything
  # for pair types without collisions, as well as updating the launch dimensions.

  for i in range(len(_COLLISION_FUNCS)):
    # we will maintain a list of number of overlaps per-pair type on GPU
    # and a base index of the first geom pair that needs to be processed

    # this will lead to a bunch of unnecessary launches, but we don't want to sync at this point
    func = _COLLISION_FUNCS[i]
    func(m, d)

##############################################################################################
# NARROW PHASE POSTPROCESSING
##############################################################################################

@wp.kernel
def create_batched_contact_arrays(
    m: Model, 
    d: Data,
    overlap_pairs: wp.array(dtype=wp.vec2i, ndim=2),
    overlap_pairs_count: wp.array(dtype=int, ndim=1),
    sort_keys: wp.array(dtype=int, ndim=2), 
    sort_indexer: wp.array(dtype=int, ndim=2), 
    contact_indexer : wp.array(dtype=int)):
    """Exports contact data in a batched structure.
    
    The contact data follows this shape convention: (batch_size, n_contacts, ...)
    
    Example structure:
    - dist: (batch_size, n_contacts)         # penetration depths
    - pos: (batch_size, n_contacts, 3)       # 3D contact positions
    - frame: (batch_size, n_contacts, 3, 3)  # contact frames (rotation matrices)
    - geom1: (batch_size, n_contacts)        # first geom indices
    - geom2: (batch_size, n_contacts)        # second geom indices
    
    Example values for a batch_size=2 with 3 contacts:
    dist = [
        [-0.01, -0.005, -0.002],  # batch 1: penetration depths
        [-0.015, -0.008, -0.003]  # batch 2: penetration depths
    ]
    pos = [
        [[1.0, 0.0, 0.0],         # batch 1: 3D positions
          [0.0, 1.0, 0.0],
          [0.5, 0.5, 0.0]],
        [[1.2, 0.1, 0.0],         # batch 2: 3D positions
          [0.1, 1.1, 0.0],
          [0.6, 0.4, 0.0]]
    ]
    """
    world_id, contact_id = wp.tid()


    #num_generated_contacts = 4 # info depending on collision type (e. g. 4 for plane_convex)

    #num_generated_contacts = overlap_pairs_count[world_id]
    #if(contact_id >= num_generated_contacts):      
    #  return

    #geom = overlap_pairs[world_id, contact_id]
    #geom1 = geom[0]
    #geom2 = geom[1]

    #condim = 3 # TODO: where to get this from?

    #base_contact_id = wp.atomic_add(contact_indexer, 0, num_generated_contacts)

    # Generate the contact(s) here
    #dist, pos, frame = plane_convex(base_contact_id)
    #plane_convex(m, d, world_id, geom1, geom2, base_contact_id, d.contact) # should write dist, pos and frame directly

    

    # write_dist_pos_frame(worldId, base_contact_id + 0, ...)
    # write_dist_pos_frame(worldId, base_contact_id + 1, ...)
    # write_dist_pos_frame(worldId, base_contact_id + 2, ...)
    # write_dist_pos_frame(worldId, base_contact_id + 3, ...)

    # Write codim
    # Write contact key (generated from g1 and g2)
    # Write contact sub id (local contact id)

    #g_min = wp.min(geom1, geom2) 
    #g_max = wp.max(geom1, geom2)
    #local_id = int(3)  
    
    # Sort key
    #base_key = (condim << 28) | (g_min << 16) | (g_max << 3) # | local_id
    #sort_keys[world_id, base_contact_id + 0] = pack_key(condim, g_min, g_max, 0) # base_key | 0
    #sort_keys[world_id, base_contact_id + 1] = pack_key(condim, g_min, g_max, 1) # base_key | 1
    #sort_keys[world_id, base_contact_id + 2] = pack_key(condim, g_min, g_max, 2) # base_key | 2
    #sort_keys[world_id, base_contact_id + 3] = pack_key(condim, g_min, g_max, 3) # base_key | 3
    
    #sort_indexer[world_id, base_contact_id + 0] = base_contact_id + 0
    #sort_indexer[world_id, base_contact_id + 1] = base_contact_id + 1
    #sort_indexer[world_id, base_contact_id + 2] = base_contact_id + 2
    #sort_indexer[world_id, base_contact_id + 3] = base_contact_id + 3

    pass

@wp.kernel
def contact_count_to_sort_sections(max_contacts_per_world: int, contact_count_per_world: wp.array(dtype=int), sort_sections_starts: wp.array(dtype=int), sort_sections_ends: wp.array(dtype=int)):
    world_id = wp.tid()

    count = contact_count_per_world[world_id]

    sort_sections_starts[world_id] = world_id * max_contacts_per_world
    sort_sections_ends[world_id] = sort_sections_starts[world_id] + count

def organize_contacts_by_constraint_dimension(
      m: Model, 
      d: Data, 
      sort_keys: wp.array(dtype=int, ndim=2), 
      sort_indexer: wp.array(dtype=int, ndim=2), 
      num_contacts_per_world : wp.array(dtype=int),
      sort_sections_starts : wp.array(dtype=int),
      sort_sections_ends : wp.array(dtype=int)):
  """Groups contacts by their contact dimension (condim).
  
  Contacts are grouped by their degrees of freedom:
  - condim=1: normal force only (frictionless)
  - condim=3: normal force + friction in 2 tangential directions
  - condim=6: normal force + friction + torsional friction
  
  Example grouping:
  condim_groups = {
      1: [sphere_plane_contacts],  # frictionless contacts
      3: [box_plane_contacts],     # friction contacts
      6: [gear_contacts]           # torsional friction contacts
  }
  
  The groups are later concatenated in order of increasing condim to ensure
  consistent contact ordering for the constraint solver.
  """

  wp.launch(
      kernel=contact_count_to_sort_sections,
      dim=d.nworld,
      inputs=[
          m.max_num_overlaps_per_world,
          num_contacts_per_world,
      ],
      outputs=[
          sort_sections_starts,
          sort_sections_ends,
      ],
  )

  wp.utils.segmented_sort_pairs_start_end(
      sort_keys.reshape(-1),
      sort_indexer.reshape(-1),
      m.max_num_overlaps_per_world * d.nworld,
      sort_sections_starts,
      sort_sections_ends,
      d.nworld,
  )



@wp.kernel
def duplicate_properties_per_contact_point(m: Model, d: Data, sort_keys: wp.array(dtype=int, ndim=2)):
  """Exports contact properties, repeating them when necessary to match contact structure.
  
  When a collision function returns multiple contact points (ncon > 1) for a single
  geometry pair, this function repeats the properties to match the number of contacts.
  
  Example:
  For a box-plane collision with 4 contact points (corners):
  Original properties:
      friction = [1.0]           # single friction value
      includemargin = [0.01]     # single margin value
  
  Repeated properties (ncon=4):
      friction = [1.0, 1.0, 1.0, 1.0]
      includemargin = [0.01, 0.01, 0.01, 0.01]
  
  This ensures each contact point has its own set of parameters, even though
  they're identical for contacts from the same geometry pair.
  """
  
  world_id, contact_id = wp.tid()
  if(contact_id >= d.contact_count[world_id]):
    return

  sort_key = sort_keys[world_id, contact_id]

  sort_key_parts = decompose_key(sort_key)
  condim = sort_key_parts[0]
  g_min = sort_key_parts[1]
  g_max = sort_key_parts[2]
  # local_id = sort_key_parts[3]

  geom1 = g_min
  geom2 = g_max
  
  eps = MJ_MINVAL

  margin = wp.maximum(m.geom_margin[geom1], m.geom_margin[geom2])
  gap = wp.maximum(m.geom_gap[geom1], m.geom_gap[geom2])
  solmix1, solmix2 = m.geom_solmix[geom1], m.geom_solmix[geom2]
  mix = solmix1 / (solmix1 + solmix2)
  mix = wp.select((solmix1 < eps) & (solmix2 < eps), mix, 0.5)  #where((solmix1 < eps) & (solmix2 < eps), 0.5, mix)
  mix = wp.select((solmix1 < eps) & (solmix2 >= eps), mix, 0.0) #where((solmix1 < eps) & (solmix2 >= eps), 0.0, mix)
  mix = wp.select((solmix1 >= eps) & (solmix2 < eps), mix, 1,0) #where(, 1.0, mix)
  # friction: max
  friction = wp.maximum(m.geom_friction[geom1], m.geom_friction[geom2])
  solref1, solref2 = m.geom_solref[geom1], m.geom_solref[geom2]
  # reference standard: mix
  solref_standard = mix * solref1 + (1 - mix) * solref2
  # reference direct: min
  solref_direct = wp.minimum(solref1, solref2)
  is_standard = (solref1[0] > 0) & (solref2[0] > 0)
  solref = wp.where(is_standard, solref_standard, solref_direct)
  # solreffriction = jp.zeros(geom1.shape + (mujoco.mjNREF,))
  # impedance: mix
  solimp = mix * m.geom_solimp[geom1] + (1 - mix) * m.geom_solimp[geom2]

  pri = m.geom_priority[geom1] != m.geom_priority[geom2]
  # if pri.any():
  if pri:
      # use priority geom when specified instead of mixing
      gp1, gp2 = m.geom_priority[geom1], m.geom_priority[geom2]
      gp = wp.select(gp1 > gp2, geom2, geom1)[pri]
      friction = friction.at[pri].set(m.geom_friction[gp])
      solref = solref.at[pri].set(m.geom_solref[gp])
      solimp = solimp.at[pri].set(m.geom_solimp[gp])

  d.contact.dist[world_id, contact_id] = margin - gap
  d.contact.friction[world_id, contact_id, 0] = friction[0] 
  d.contact.friction[world_id, contact_id, 1] = friction[0]
  d.contact.friction[world_id, contact_id, 2] = friction[1]
  d.contact.friction[world_id, contact_id, 3] = friction[2]
  d.contact.friction[world_id, contact_id, 4] = friction[2]
  d.contact.solref[world_id, contact_id] = solref
  d.contact.solimp[world_id, contact_id] = solimp
  d.contact.includemargin[world_id, contact_id] = margin
  d.contact.dim[world_id, contact_id] = condim
  d.contact.geom[world_id, contact_id, 0] = geom1
  d.contact.geom[world_id, contact_id, 1] = geom2
  d.contact.solreffriction[world_id, contact_id] = wp.vector(wp.float32(0.0), length=MJ_NREF)

# TODO: Not strictly required for now. Implement later
def select_deepest_penetrating_contacts():
    """Limits the number of contacts per condim group based on penetration depth.
    
    When max_contact_points is set, this function:
    1. Processes each condim group separately
    2. Selects the contacts with deepest penetration
    3. Discards less important contacts
    
    Example:
    With max_contact_points = 2 and these penetration depths:
        dist = [-0.05, -0.03, -0.01, -0.02]
    
    Result after filtering:
        dist = [-0.05, -0.03]  # kept two deepest penetrations
    
    This is important because:
    1. Reduces simulation computational cost
    2. Keeps the most physically significant contacts
    3. Maintains contact type diversity by limiting each condim group separately
    
    Example with max_contact_points = 4:
    - 6 contacts with condim=1 → reduced to 4
    - 3 contacts with condim=3 → stays 3
    - 5 contacts with condim=6 → reduced to 4
    Final result: 11 total contacts (4 + 3 + 4)
    """
    pass

def contacts_to_world_postprocess(m: Model, d: Data) -> Data:
  # takes the contact pairs for each pair type, and
  # - create the batched contact arrays (per-world)
  # - organize them by constraint dimension
  # - duplicate properties for contacts with multiple points

  wp.launch(
    kernel=create_batched_contact_arrays,
    dim=(d.nworld, m.max_num_overlaps_per_world),
    inputs=[
      m,
      d,
      d.broadphase_pairs,
      d.result_count,
      d.narrow_phase_sort_keys,
      d.narrow_phase_sort_indexer,
      d.narrow_phase_contact_indexer,
    ],
  )
  
  organize_contacts_by_constraint_dimension(m, d, d.narrow_phase_sort_keys, d.narrow_phase_sort_indexer, d.narrow_phase_contact_count, 
    d.narrow_phase_sort_sections_starts, d.narrow_phase_sort_sections_ends)
  
  wp.launch(
    kernel=duplicate_properties_per_contact_point,
    dim=(d.nworld, d.max_num_contacts_per_world),
    inputs=[m, d, d.narrow_phase_sort_keys],
  )
  
  select_deepest_penetrating_contacts(m, d, d.narrow_phase_sort_keys, d.narrow_phase_sort_indexer, d.narrow_phase_sort_sections) # Optional
  

def collision(m: Model, d: Data) -> Data:
  """Collision detection."""

  # do we need to have a stage here for the specific per-pair collisions?
  # is it allowed to have per-pair + global collisions?
  # maybe we can preprocess that and handle it in put_model?

  broad_phase(m, d) # per-world
  filtering(m, d) # drop unwanted contacts - take advantage of the sorting putting invalid keys at the end.
  overlaps_to_type_buckets(m, d) # per-world -> per-pair type
  narrow_phase(m, d) # per-pair type
  contacts_to_world_postprocess(m, d) # per-pair type -> per-world
