import os
import xml.etree.ElementTree as ET
import torch
import torchvision as tv
from torchvision import tv_tensors
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
