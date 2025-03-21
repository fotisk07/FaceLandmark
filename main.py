import torch
import torchvision.transforms.v2 as v2

from src.dataset import FaceDataset
from src.models import Baseline

xml_file_train = "data/dlib_faces_5points/train_cleaned.xml"
xml_file_test = "data/dlib_faces_5points/test_cleaned.xml"
root_dir = "data"
batch_size = 1
input_size = 3 * 256 * 256
lr = 4e-3
epochs = 1

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    @torch.no_grad()
    def eval_model():
        model.eval()
        total_loss = 0
        for sample in val_loader:
            output, loss = model(sample["image"], sample["bbox"])
            total_loss += loss.item()

        return total_loss / len(val_loader)

    transform = v2.Compose(
        [
            v2.Resize((256, 256)),
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
        ]
    )
    train_dataset = FaceDataset(
        xml_file=xml_file_train, root_dir=root_dir, transform=transform
    )
    val_dataset = FaceDataset(
        xml_file=xml_file_train, root_dir=root_dir, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    model = Baseline(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        val_loss = eval_model()
        print(f"Starting training, val loss : {val_loss:4f}")

        total_train_loss = 0
        for step, sample in enumerate(train_loader):
            optimizer.zero_grad()
            output, loss = model(sample["image"].to(device), sample["bbox"].to(device))
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f"Epoch {epoch:2d} |Step {step:4d} | Loss : {loss.item():.3f}")

        val_loss = eval_model()
        train_loss = total_train_loss / len(train_loader)
        print(f"----- Finished Epoch {epoch+1} --------")
        print(f"Train loss : {train_loss:4f} | Val loss: {val_loss:4f}")
        print("----------------------------------------")


if __name__ == "__main__":
    main()
