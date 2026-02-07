import torch
import torch.nn.functional as F

def simclr_loss(p1: torch.Tensor, p2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    B = p1.size(0)
    z = torch.cat([p1, p2], dim=0).float()  # (2B, D)
    sim = (z @ z.t()) / float(temperature)
    sim.fill_diagonal_(-1e4)
    pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(sim.device)
    return F.cross_entropy(sim, pos)

def temporal_consistency(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    # 1 - cosine similarity, averaged
    return (1.0 - (z1 * z2).sum(dim=-1)).mean()
