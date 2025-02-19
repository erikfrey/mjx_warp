import warp as wp

from .types import Model
from .types import Data


def collision(m: Model, d: Data):
  """Main collision function."""

  @wp.kernel
  def _root(m: Model, d: Data):
    d.ncon_total[0] = 0
  
  wp.launch(_root, dim=(1,), inputs=[m, d])



def broadphase(m: Model, d: Data):
  """Broadphase collision detector."""
  # TODO(team): sweep and prune instead of n^2

