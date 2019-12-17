from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset
}
