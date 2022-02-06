from .bbox_heads import DIISeriesHead, DIIIconHead
from .sparse_series_roi_head import SparseSeriesRoIHead
from .sparse_inconsistent_roi_head import SparseInconRoIHead
from .sparse_series_roi_head_undetach import SparseSeriesRoIHeadUnDetach
from .sparse_series_roi_head_undetach0stage import SparseSeriesRoIHeadUndetach0stage

__all__ = [
    'DIIIconHead', 'DIISeriesHead', 'SparseSeriesRoIHead', 'SparseInconRoIHead', 'SparseSeriesRoIHeadUnDetach','SparseSeriesRoIHeadUndetach0stage'
]
