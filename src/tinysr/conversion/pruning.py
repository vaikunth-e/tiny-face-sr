"""True dense channel removal for the baseline topology; no sparse-kernel claims."""


def prune_model(model, keep):
    """Rank each feature layer by filter L1, then remove its output and successor inputs."""
    import torch
    from tinysr.models.residual import TinySR

    width = model.channels
    if (
        not isinstance(keep, int)
        or not 4 <= keep < width
        or model.blocks != 2
        or model.scale != 4
    ):
        raise ValueError(
            "Expected a smaller width >=4 for the historical two-block 4x model"
        )
    student = TinySR(variant="residual", channels=keep, blocks=2, scale=4).to(
        next(model.parameters()).device
    )
    indices = []
    with torch.no_grad():
        for layer in (0, 2, 4):
            scores = model.network[layer].weight.abs().sum(dim=(1, 2, 3))
            selected = (
                torch.argsort(scores, descending=True, stable=True)[:keep].sort().values
            )
            indices.append(selected)
        previous = torch.arange(3, device=indices[0].device)
        for layer, selected in zip((0, 2, 4), indices):
            source, target = model.network[layer], student.network[layer]
            target.weight.copy_(source.weight[selected][:, previous])
            target.bias.copy_(source.bias[selected])
            previous = selected
        student.network[6].weight.copy_(model.network[6].weight[:, previous])
        student.network[6].bias.copy_(model.network[6].bias)
    return student, [i.cpu().tolist() for i in indices]
