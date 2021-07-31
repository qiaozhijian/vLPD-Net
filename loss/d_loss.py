import torch

from misc.log import log_string


def make_d_loss(params):
    if params.d_loss == 'BCELogitWithMask':
        loss_fn = BCELogitWithMask()
    elif params.d_loss == 'MSEWithMask':
        loss_fn = MSEWithMask()
    elif params.d_loss == 'MSE':
        loss_fn = MSE()
    elif params.d_loss == 'BCELogit':
        loss_fn = BCELogit()
    else:
        log_string('Unknown loss: {}'.format(params.d_loss))
        raise NotImplementedError
    return loss_fn


class DisLossBase:
    # Hard triplet miner
    def __init__(self):
        self.distance = None
        self.mask = False
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def __call__(self, syn, hard_triplets, real=None):
        assert self.distance is not None
        assert len(syn.shape) == 2

        if self.mask:
            a, p, n = hard_triplets
            syn = syn[a]
            real = real[a] if real is not None else None

        if real is not None:
            assert len(real.shape) == 2
            syn_label = torch.zeros((syn.shape[0], 1), dtype=torch.float32, device=self.device)
            real_label = torch.ones((real.shape[0], 1), dtype=torch.float32, device=self.device)
            label = torch.cat((syn_label, real_label), dim=1)
            out = torch.cat((syn, real), dim=1)
        else:
            label = torch.ones((syn.shape[0], 1), dtype=torch.float32, device=self.device)
            out = syn

        loss = self.distance(out, label)

        if real is not None:
            stats = {'d_loss': loss.detach().cpu().item()}
        else:
            stats = {'g_d_loss': loss.detach().cpu().item()}

        return loss, stats


class BCELogitWithMask(DisLossBase):
    def __init__(self):
        super(BCELogitWithMask, self).__init__()
        self.distance = torch.nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
        self.mask = True


class MSEWithMask(DisLossBase):
    def __init__(self):
        super(MSEWithMask, self).__init__()
        self.distance = torch.nn.MSELoss(reduction='mean').to(self.device)
        self.mask = True


class BCELogit(DisLossBase):
    def __init__(self):
        super(BCELogit, self).__init__()
        self.distance = torch.nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
        self.mask = False


class MSE(DisLossBase):
    def __init__(self):
        super(MSE, self).__init__()
        self.distance = torch.nn.MSELoss(reduction='mean').to(self.device)
        self.mask = False
