import os
import xml.etree.ElementTree as ET
import torch
import torchvision as tv
from torchvision import tv_tensors
import matplotlib.pyplot as plt
import numpy as np


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
        image = tv.io.decode_image(sample["img_path"])
        bbox = tv_tensors.BoundingBoxes(
            sample["bbox"], canvas_size=image.shape[-2::], format="XYXY"
        )
        keypoints = torch.tensor(
            list(sample["keypoints"].values()), dtype=torch.float32
        )

        if self.transform:
            image = self.transform(image)

        return {"image": image, "bbox": bbox, "keypoints": keypoints}


def visualize_sample(sample):
    image = sample["image"]
    if isinstance(image, torch.Tensor):
        image = tv.transforms.ToPILImage()(image)

    # Convert image to numpy for visualization
    image_np = np.array(image)

    # Draw bounding box
    bbox = sample["bbox"][0].tolist()

    # Plotting the image and bounding box
    fig, ax = plt.subplots()
    ax.imshow(image_np)
    ax.add_patch(
        plt.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
    )

    # Draw keypoints
    for point in sample["keypoints"].tolist():
        ax.plot(point[0], point[1], "bo")  # Blue circles for keypoints

    ax.axis("off")
    return fig, ax
