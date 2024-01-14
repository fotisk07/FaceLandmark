from src.dataset import create_data
from src.Training.training import Trainer
from src.Training import losses
from src.models.baseline import Baseline
import torch
import wandb 
import src.Training.evaluation as evaluation
from src.Training.evaluation import Evaluator
from cliconfig import make_config



if __name__ == "__main__":
    config = make_config("configs/default.yaml").dict

    lr = config["lr"]
    batch_size = config["batch_size"]
    csv_file_path = config["csv_file_path"]
    root_dir = config["root_dir"]
    evaluate_every = config["evaluate_every"]
    save_model = config["save_model"]
    save_every = config["save_every"]
    num_workers = config["num_workers"]
    epochs = config["epochs"]
    verbose = config["verbose"]
    log_wandb = config["wandb"]
    cuda = config["cuda"]



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if cuda else 'cpu'

    if log_wandb:
        run = wandb.init(project="Face-Landmarks")

    train_loader, valid_loader = create_data(csv_file_path, root_dir, batch_size, num_workers)

    model = Baseline(10, name = config["model_name"], gen = config["model_gen"]).to(device)
    landmark_loss = losses.landmark_loss
    mask_loss = losses.mask_loss
    evaluator = evaluation.Evaluator(model, mask_loss, landmark_loss, device=device)


    if config["train"]:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = Trainer(model, optimizer, mask_loss, landmark_loss, device=device, 
                    log_wandb=log_wandb, save=save_model, save_every=save_every, 
                    evaluator=evaluator)

        trainer.train(train_loader, valid_loader, 
                        epochs, verbose=verbose, 
                        evaluate_every=evaluate_every)

    if config["predict"]:
        print("Predicting")

        _,valid_loader = create_data(csv_file_path, root_dir, batch_size, num_workers, 
                                     only_valid=True)
        model_dict = torch.load(config["model_dir"], map_location=torch.device('cpu'))
        model.load_state_dict(model_dict['model_state_dict'])
        model.eval()

        save_figures_path = config["save_figures_path"]
        evaluator.create_img_landmarks_graph(valid_loader, num_images=4, show=False,
                                              save=True, save_path=save_figures_path)
        evaluator.create_landmark_comparaison_graph(valid_loader, num_images=4, show=False,
                                                     save=True, save_path=save_figures_path)
        evaluator.create_mask_comparaison_graph(valid_loader, num_images=4, show=False, save=True,
                                                 save_path=save_figures_path)
