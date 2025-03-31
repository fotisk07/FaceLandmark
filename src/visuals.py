import numpy as np
import torch
import torchvision as tv
import matplotlib.pyplot as plt

tensor = torch.Tensor
array = np.ndarray


def visualize_grid(samples: list[dict], pred_keypoints: array, cols: int = 4):

    rows = len(samples) // cols + (1 if len(samples) % cols else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()

    for i, sample in enumerate(samples):
        ax = axes[i]
        plot_image_with_predictions(ax, sample, pred_keypoints[i])

    for j in range(len(samples), len(axes)):
        axes[j].axis("off")

    return fig, axes


def plot_image_with_predictions(ax, sample: dict | tensor, pred_keypoints: array):

    image = sample["image"]
    if isinstance(image, torch.Tensor):
        image = tv.transforms.ToPILImage()(image)
    image_np = np.array(image)
    bbox = sample["bbox"]
    face_box = bbox[0].tolist()
    gt_keypoints = bbox[1:].numpy()[:, :2]
    pred_keypoints = pred_keypoints[:, :2]

    ax.imshow(image_np)

    # Draw face bounding box
    ax.add_patch(
        plt.Rectangle(
            (face_box[0], face_box[1]),
            face_box[2] - face_box[0],
            face_box[3] - face_box[1],
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
    )

    # Plot ground truth keypoints
    ax.scatter(
        gt_keypoints[:, 0],
        gt_keypoints[:, 1],
        c="blue",
        marker="o",
        label="Ground Truth",
    )

    # Plot predicted keypoints
    ax.scatter(
        pred_keypoints[:, 0],
        pred_keypoints[:, 1],
        c="green",
        marker="x",
        label="Predicted",
    )

    # Add keypoint numbers
    for i, (gt, pred) in enumerate(zip(gt_keypoints, pred_keypoints)):
        ax.text(gt[0], gt[1], str(i), color="blue", fontsize=8, fontweight="bold")
        ax.text(pred[0], pred[1], str(i), color="green", fontsize=8, fontweight="bold")

    ax.axis("off")
    ax.legend()
