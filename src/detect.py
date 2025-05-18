import torch
# DEMO FUNC
def participation_ratio_of_model(model: torch.nn.Module) -> float:
    """Compute the participation ratio of all *trainable* parameters."""
    # flatten parameters into one long vector
    flat = torch.cat(
        [p.data.reshape(-1).float() for p in model.parameters() if p.requires_grad]
    )
    s2 = torch.sum(flat.pow(2))
    s4 = torch.sum(flat.pow(4))
    return (s2.pow(2) / s4).item()
