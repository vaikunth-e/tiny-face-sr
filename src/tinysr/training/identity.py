
import torch
import torch.nn.functional as F


def prepare_for_facenet(images):
    images = images.clamp(0, 1)

    images = F.interpolate(
        images,
        size=(160, 160),
        mode="bilinear",
        align_corners=False
    )

    return (images * 255.0 - 127.5) / 128.0


def identity_loss(prediction, target, face_model):
    face_model.eval()

    # Prediction: gradients must flow back into TinySR
    sr_embedding = face_model(
        prepare_for_facenet(prediction)
    )

    # Ground truth: no gradients needed
    with torch.no_grad():
        hr_embedding = face_model(
            prepare_for_facenet(target)
        )

    similarity = F.cosine_similarity(
        sr_embedding,
        hr_embedding,
        dim=1
    )

    return (1 - similarity).mean()
