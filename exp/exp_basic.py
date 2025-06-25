import os
import torch
from models import TimeMixer, FBM_S,FBP_NL,TimesNet,FiLM,iTransformer,FBP_L,FBP_NP, NLinear,PatchTST


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimeMixer': TimeMixer,
            'FBM-Super': FBM_S,
            'FBM-NL': FBP_NL,
            'FBM-L': FBP_L,
            'FBM-NP': FBP_NP,                       
            'TimesNet': TimesNet,
            'FiLM': FiLM,
            'NLinear': NLinear,
            'iTransformer': iTransformer,
            'PatchTST': PatchTST,

        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            import platform
            if platform.system() == 'Darwin':
                device = torch.device('mps')
                print('Use MPS')
                return device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            if self.args.use_multi_gpu:
                print('Use GPU: cuda{}'.format(self.args.device_ids))
            else:
                print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
