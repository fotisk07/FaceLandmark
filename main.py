import torch
import torchvision.transforms.v2 as v2
import hydra
from pathlib import Path
from omegaconf import DictConfig

from src.dataset import FaceDataset
from src.models import Model, model_dict

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

hydra_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)


@torch.no_grad()
def eval_model(model: torch.nn.Module, dataloaders: tuple, eval_iters: int) -> dict:
    model.eval()
    splits = ["train", "val"]
    avrg_loss = {s: 0.0 for s in splits}  # Initialize with float 0.0

    for i, split in enumerate(splits):
        loader = iter(dataloaders[i])
        total_loss_split = 0.0
        batches_processed = 0
        for _ in range(eval_iters):
            try:
                sample = next(loader)
                # Assuming keys "image", "bbox" and global 'device'
                image, targets = sample["image"].to(device), sample["bbox"].to(device)
                output, loss = model(image, targets)
                total_loss_split += loss.item()
                batches_processed += 1
            except StopIteration:
                print(
                    f"  Info: Ran out of data in '{split}' split after"
                    f"{batches_processed} batches."
                )
                break
        # Calculate average for the split using actual batches processed
        if batches_processed > 0:
            avrg_loss[split] = total_loss_split / batches_processed
        else:
            print(f"  Warning: No batches were processed for '{split}' split.")
            avrg_loss[split] = float("nan")  # Or 0.0, based on preference

    model.train()
    return avrg_loss


def save(model: Model, epoch: int):
    path = hydra_dir / "models" / f"epoch_{epoch}"
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
        },
        path,
    )


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    g = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    transform = v2.Compose(
        [
            v2.Resize((256, 256)),
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
        ]
    )
    train_loader = torch.utils.data.DataLoader(
        FaceDataset(
            xml_file=cfg.xml_file_train, root_dir=cfg.root_dir, transform=transform
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = torch.utils.data.DataLoader(
        FaceDataset(
            xml_file=cfg.xml_file_test, root_dir=cfg.root_dir, transform=transform
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=g,
    )

    model: Model = model_dict[cfg.model](input_size=cfg.input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print("Started Training")
    eval_losses_per_epoch = []
    for epoch in range(cfg.epochs):
        total_steps = len(train_loader)
        for step, sample in enumerate(train_loader):
            optimizer.zero_grad()
            output, loss = model(sample["image"].to(device), sample["bbox"].to(device))
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                avg_loss = eval_model(model, (train_loader, val_loader), cfg.eval_iters)
                train_loss, eval_loss = avg_loss["train"], avg_loss["val"]
                print(
                    f"Epoch {epoch:02d} | Step {step:04d}/{total_steps} | "
                    f"Train Loss: {train_loss:.3f} | Val Loss: {eval_loss:.3f}"
                )
        # Calculating epoch-wise stats
        avg_loss = eval_model(model, (train_loader, val_loader), cfg.eval_iters)
        train_loss, eval_loss = avg_loss["train"], avg_loss["val"]
        eval_losses_per_epoch.append(eval_loss)
        if eval_loss <= min(eval_losses_per_epoch):
            save(model, epoch)

        print(f"----- Finished Epoch {epoch+1} --------")
        print(f"Train Loss: {train_loss:.3f} | Val Loss: {eval_loss:.3f}")
        print("----------------------------------------")


if __name__ == "__main__":
    main()
