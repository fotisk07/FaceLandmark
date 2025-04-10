import torch
import hydra
from pathlib import Path
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt

from src.models import Model
from src.visuals import visualize_grid

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")


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
                output, loss = model({k: v.to(device) for k, v in sample.items()})
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
    hydra_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    dir = hydra_dir / "models"
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / f"epoch_{epoch}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch + 1,
        },
        path,
    )


@torch.no_grad()
def sample_model(model, loader, num_visuals, cols, epoch=0):
    idxs = np.random.choice(len(loader.dataset), num_visuals)
    samples = [loader.dataset[i] for i in idxs]
    pred_keypoints = model.generate(
        torch.stack([sample["image"] for sample in samples]).to(device)
    )
    fig, ax = visualize_grid(samples, pred_keypoints, cols=cols)
    path = (
        Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        / "sampling"
        / f"grid_{epoch}.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    g = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    transform = hydra.utils.call(cfg.transf)
    train_dataset = hydra.utils.instantiate(cfg.train_data, transform=transform)
    val_dataset = hydra.utils.instantiate(cfg.val_data)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=g,
    )

    model: Model = hydra.utils.instantiate(cfg.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print("Started Training")
    eval_losses_per_epoch = []
    for epoch in range(cfg.epochs):
        total_steps = len(train_loader)
        for step, sample in enumerate(train_loader):
            optimizer.zero_grad()
            output, loss = model({k: v.to(device) for k, v in sample.items()})
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

        print(f"--------- Finished Epoch {epoch+1} ---------------")
        print(f"Train Loss: {train_loss:.3f} | Val Loss: {eval_loss:.3f}")
        if cfg.sample:
            sample_model(model, val_loader, cfg.num_visuals, cfg.cols, epoch)
            print("Samples saved ! ")
        print("----------------------------------------")


if __name__ == "__main__":
    main()
