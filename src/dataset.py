import os
import xml.etree.ElementTree as ET
import torch
import torchvision as tv
from torchvision import tv_tensors
import matplotlib.pyplot as plt
import numpy as np

tensor = torch.Tensor
array = np.ndarray


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, xml_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.parse_xml(xml_file)

    def parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        data = []

        for image in root.findall("images/image"):
            img_path = os.path.join(
                self.root_dir, "dlib_faces_5points", image.get("file")
            )
            # width, height = int(image.get("width")), int(image.get("height"))

            for box in image.findall("box"):
                bbox = [
                    int(box.get("left")),
                    int(box.get("top")),
                    int(box.get("left")) + int(box.get("width")),
                    int(box.get("top")) + int(box.get("height")),
                ]
                keypoints = {}
                for part in box.findall("part"):
                    keypoints[part.get("name")] = (
                        int(part.get("x")),
                        int(part.get("y")),
                    )

                data.append(
                    {"img_path": img_path, "bbox": bbox, "keypoints": keypoints}
                )

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = tv.io.read_image(
            sample["img_path"]
        )  # Use read_image instead of decode_image
        face_box = torch.tensor(sample["bbox"], dtype=torch.float32).unsqueeze(
            0
        )  # Shape: (1, 4)

        keypoints = torch.tensor(
            list(sample["keypoints"].values()), dtype=torch.float32
        )  # Shape: (5, 2)

        # Convert keypoints to (x, y, x, y) format with zero width/height
        keypoint_boxes = torch.cat([keypoints, keypoints], dim=1)  # Shape: (5, 4)

        # Stack face box and keypoints into one BoundingBoxes object
        all_boxes = torch.cat([face_box, keypoint_boxes], dim=0)  # Shape: (6, 4)

        bbox = tv_tensors.BoundingBoxes(
            all_boxes, canvas_size=image.shape[-2:], format="XYXY"
        )

        if self.transform:
            image, bbox = self.transform(image, bbox)

        return {"image": image, "bbox": bbox}


def visualize_sample(sample):
    image = sample["image"]
    if isinstance(image, torch.Tensor):
        image = tv.transforms.ToPILImage()(image)

    # Convert image to numpy for visualization
    image_np = np.array(image)

    # Bounding boxes (first one is face, rest are keypoints)
    bbox = sample["bbox"]
    face_box = bbox[0].tolist()  # First bbox is the face
    keypoints = bbox[1:].numpy()[:, :2]  # Extract (x, y) for keypoints

    # Plot image
    fig, ax = plt.subplots()
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

    # Draw keypoints
    ax.scatter(keypoints[:, 0], keypoints[:, 1], c="blue", marker="o")  # Blue circles

    ax.axis("off")
    return fig, ax


def visualize_predictions(
    image: torch.Tensor, gt_keypoints: torch.Tensor, pred_keypoints: torch.Tensor | list
):
    # Unbatch if batch
    if len(image.shape) == 4:
        image = image[0]
    if len(gt_keypoints.shape) == 3:
        gt_keypoints = gt_keypoints[0]
    if len(pred_keypoints.shape) == 3:
        pred_keypoints = pred_keypoints[0]

    image = image.permute(1, 2, 0).detach().cpu().numpy()

    # Extract bounding box and keypoints
    face_box = gt_keypoints[0].tolist()
    gt_keypoints = (
        gt_keypoints[1:].detach().cpu().numpy()[:, :2]
    )  # Ground truth keypoints

    if isinstance(pred_keypoints, torch.Tensor):
        pred_keypoints = (
            pred_keypoints.detach().cpu().numpy()[:, :2]
        )  # Predicted keypoints

    # Plot image
    fig, ax = plt.subplots()
    ax.imshow(image)

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

    # Plot keypoints with numbers
    for i, (gt, pred) in enumerate(zip(gt_keypoints, pred_keypoints)):
        ax.scatter(
            gt[0], gt[1], c="blue", marker="o", label="Ground Truth" if i == 0 else ""
        )
        ax.scatter(
            pred[0], pred[1], c="green", marker="x", label="Predicted" if i == 0 else ""
        )
        ax.text(gt[0], gt[1], str(i), color="blue", fontsize=8, fontweight="bold")
        ax.text(pred[0], pred[1], str(i), color="green", fontsize=8, fontweight="bold")

    ax.legend()
    ax.axis("off")
    return fig, ax
