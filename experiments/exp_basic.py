import os
import torch
from model import (
    Transformer,
    Informer,
    Reformer,
    Flowformer,
    Flashformer,
    iTransformer,
    iInformer,
    iReformer,
    iFlowformer,
    iFlashformer,
)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "Transformer": Transformer,
            "Informer": Informer,
            "Reformer": Reformer,
            "Flowformer": Flowformer,
            "Flashformer": Flashformer,
            "iTransformer": iTransformer,
            "iInformer": iInformer,
            "iReformer": iReformer,
            "iFlowformer": iFlowformer,
            "iFlashformer": iFlashformer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_mps:
            os.environ["mps_VISIBLE_DEVICES"] = (
                str(self.args.mps) if not self.args.use_multi_mps else self.args.devices
            )
            device = torch.device("mps:{}".format(self.args.mps))
            print("Use mps: mps:{}".format(self.args.mps))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
