import os
import xml.etree.ElementTree as ET
import torch
import torchvision as tv
from torchvision import tv_tensors
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as F
import numpy as np

tensor = torch.Tensor
array = np.ndarray


# ------------------ My transforms -------------------------
class FaceCrop(torch.nn.Module):
    def __init__(self, padding: float = 0):
        super().__init__()
        self.padding = padding

    def __call__(self, img, box):
        img, bbox = super().__call__(img, box)

        # Extract first bounding box for cropping
        x1, y1, x2, y2 = bbox[0].tolist()[:4]

        # Add padding around the face
        padding_x = (x2 - x1) * self.padding  # padding
        padding_y = (y2 - y1) * self.padding  # padding

        # Compute new bounding box with padding
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(img.shape[2], x2 + padding_x)
        y2 = min(img.shape[1], y2 + padding_y)

        # Crop the image to the padded bounding box
        cropped_img = F.crop(img, int(y1), int(x1), int(y2 - y1), int(x2 - x1))

        # Resize cropped image back to original size
        img = F.resize(cropped_img, (img.shape[1], img.shape[2]))

        # Adjust remaining bounding boxes to match resized image
        scale_x = img.shape[2] / (x2 - x1)
        scale_y = img.shape[1] / (y2 - y1)
        bbox[:, 0] = (bbox[:, 0] - x1) * scale_x
        bbox[:, 1] = (bbox[:, 1] - y1) * scale_y
        bbox[:, 2] = (bbox[:, 2] - x1) * scale_x
        bbox[:, 3] = (bbox[:, 3] - y1) * scale_y

        return img, bbox


# ----------------------Composed pretransforms -----------------------------
def baseTransform(size=(256, 256)):
    return v2.Compose(
        [
            v2.Resize(size),
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
        ]
    )


def advancedTransform(size, **kwargs):
    return v2.Compose(
        [
            v2.Resize(size),
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.RandomHorizontalFlip(kwargs.get("p_hflip", 0.5)),
            v2.ColorJitter(
                brightness=kwargs.get("brightness", 0.2),
                saturation=kwargs.get("saturation", 0.1),
                contrast=kwargs.get("contrast", 0.2),
                hue=kwargs.get("hue", 0.1),
            ),
            v2.RandomPerspective(
                distortion_scale=kwargs.get("scale_distort", 0.3),
                p=kwargs.get("p_distort", 0.5),
            ),
        ]
    )


# ----------------------Datasets -----------------------------
class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, xml_file, root_dir, transform=baseTransform()):
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
