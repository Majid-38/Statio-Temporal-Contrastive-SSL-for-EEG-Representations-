import torch

@torch.no_grad()
def aug_view(
    x: torch.Tensor,
    jitter_sigma: float = 0.015,
    scale_std: float = 0.08,
    time_mask_frac: float = 0.06,
    channel_drop_prob: float = 0.08,
) -> torch.Tensor:
    """
    Physiologically consistent augmentations for EEG windows.
    x: (L, C) standardized.
    """
    # additive jitter
    if jitter_sigma > 0:
        x = x + torch.randn_like(x) * jitter_sigma

    # channel-wise scaling
    if scale_std > 0:
        s = torch.randn(1, x.shape[1], device=x.device) * scale_std + 1.0
        x = x * s

    # temporal mask
    if time_mask_frac > 0:
        m = int(x.shape[0] * time_mask_frac)
        if m > 1:
            st = torch.randint(0, max(1, x.shape[0] - m), (1,), device=x.device).item()
            x[st:st+m, :] = 0.0

    # channel dropout
    if channel_drop_prob > 0:
        drop = (torch.rand(x.shape[1], device=x.device) < channel_drop_prob)
        if drop.any():
            x[:, drop] = 0.0

    return x
