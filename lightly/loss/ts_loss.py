import torch
from torch.nn.functional import cosine_similarity
from lightly.loss.gather import GatherLayer

softmax = torch.nn.Softmax(dim=1)

def sharpen(p, T = 0.25):
    sharp_p = p**(1./T)
    sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
    return sharp_p

def snn(query, supports, gather_supports, tau = 0.1):
    # Step 1: normalize embeddings
    query = torch.nn.functional.normalize(query)
    supports = torch.nn.functional.normalize(supports)

    # Step 2: gather embeddings from all workers
    if gather_supports:
        #print("TODO: implement distributed init: https://pytorch.org/docs/master/notes/ddp.html#example")
        #exit(1)
        supports = torch.cat(GatherLayer.apply(supports), 0)

    #print(torch.sum(query @ supports.T, dim=0).shape, (query @ su
    mtx = query @ supports.T
    M,N = mtx.shape[0], mtx.shape[1]
    #mtx = mtx[:(M//2)]+mtx[(M//2):]
    mtx = mtx[:,0*(N//8):(1)*(N//8)]+mtx[:,1*(N//8):(2)*(N//8)]+mtx[:,2*(N//8):(3)*(N//8)]
    #mtx = torch.sum(*[mtx[:,i*(N//8):(i+1)*(N//8)] for i in range(8)])
    return softmax(mtx / tau)

class TsLoss(torch.nn.Module):
    def __init__(self, gather_supports, eps: float = 1e-4) -> None:
        """Same parameters as in torch.nn.CosineSimilarity

        Args:
            dim (int, optional):
                Dimension where cosine similarity is computed. Default: 1
            eps (float, optional):
                Small value to avoid division by zero. Default: 1e-8
        """
        super().__init__()
        self.eps = eps # Xentloss -> 1e-8
        self.gather_supports = gather_supports

    def forward(self, z: torch.Tensor, p: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        #multicrop = 0
        #batch_size = len(z) // (2+multicrop)

        probs = snn(p, supp, self.gather_supports)

        # Step 2: compute targets for anchor predictions
        # ...
        with torch.no_grad():
            targets = snn(z, supp.detach(), self.gather_supports)
            targets = sharpen(targets)
            #if multicrop > 0:
            #    mc_target = 0.5*(targets[:batch_size]+targets[batch_size:])
            #    targets = torch.cat([targets, *[mc_target for _ in range(multicrop)]], dim=0)
            targets[targets < self.eps] *= 0  # numerical stability

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = torch.sum(torch.log(probs**(-targets)), dim=1).mean()
        return loss