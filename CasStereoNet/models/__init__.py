from models.gwcnet import GwcNet
from models.loss import model_psmnet_loss, stereo_psmnet_loss
from models.loss import model_gwcnet_loss
from models.psmnet import PSMNet

__models__ = {
    "gwcnet": GwcNet,
    "gwcnet-c": PSMNet
}

__loss__ = {
    "gwcnet": stereo_psmnet_loss,
    "gwcnet-c": stereo_psmnet_loss
}