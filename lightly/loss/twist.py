import torch
from torch.nn.functional import cosine_similarity
from lightly.loss.gather import GatherLayer

EPS = 1e-6

class EntLoss(torch.nn.Module):
    def __init__(self, lam1, lam2, world_size, pqueue=None):
        super(EntLoss, self).__init__()
        self.lam1 = lam1
        self.lam2 = lam2
        self.pqueue = pqueue
        self.world_size = world_size
    
    def forward(self, feat1, feat2, use_queue=False):
        tau_kl = 1.0
        tau = 1.0
        probs1 = torch.nn.functional.softmax(feat1/tau_kl, dim=-1)
        probs2 = torch.nn.functional.softmax(feat2/tau_kl, dim=-1)
        loss = dict()
        loss['kl'] = 0.5 * (KL(probs1, probs2) + KL(probs2, probs1))

        sharpened_probs1 = torch.nn.functional.softmax(feat1/tau, dim=-1)
        sharpened_probs2 = torch.nn.functional.softmax(feat2/tau, dim=-1)
        #sharpened_probs2 = sharpen(sharpened_probs2)
        loss['eh'] = 0.5 * (EH(sharpened_probs1) + EH(sharpened_probs2))

        # whether use historical data
        loss['he'] = 0.5 * (HE(sharpened_probs1, self.world_size) + HE(sharpened_probs2, self.world_size))

        loss['final'] = loss['kl'] + ((1+self.lam1)*loss['eh'] - self.lam2*loss['he'])
        return loss['final']

def KL(probs1, probs2):
    kl = (probs1 * (probs1 + EPS).log() - probs1 * (probs2 + EPS).log()).sum(dim=1)
    kl = kl.mean()
    torch.distributed.all_reduce(kl)
    return kl

def CE(probs1, probs2):
    #ce = - (((probs2 + args.EPS)**probs1).log()).sum(dim=1)
    ce = - (probs1 * (probs2 + EPS).log()).sum(dim=1)
    ce = ce.mean()
    torch.distributed.all_reduce(ce)
    return ce

def HE(probs, world_size):
    mean = probs.mean(dim=0)
    torch.distributed.all_reduce(mean)
    ent  = - (mean * (mean + world_size * EPS).log()).sum()
    return ent

def EH(probs):
    ent = - (probs * (probs + EPS).log()).sum(dim=1)
    mean = ent.mean()
    torch.distributed.all_reduce(mean)
    return mean