import hydra
import lightning as L
import numpy as np
import pickle
import time
import torch
import torch.nn.functional as F
import os

from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from scRatio.datamodules import ArrayDataset
from scRatio.models import *

def prepare_dataset(n, N, cond_dim, locs):
    C = np.random.randint(low=0, high=cond_dim, size=(N))
    X = np.concatenate([np.random.normal(loc=locs[c], scale=1, size=(1, n)) for c in C])

    X_train, X_val_test, C_train, C_val_test = train_test_split(X, C, test_size=20_000)
    X_val, X_test, C_val, C_test = train_test_split(X_val_test, C_val_test, test_size=10_000)

    X_train = torch.tensor(X_train).to("cuda").float()
    C_train = F.one_hot(torch.tensor(C_train).long(), num_classes=cond_dim).to("cuda").float()

    X_val = torch.tensor(X_val).to("cuda").float()
    C_val = F.one_hot(torch.tensor(C_val).long(), num_classes=cond_dim).to("cuda").float()
    return X_train, X_val, C_train, C_val

def build_train_loader(X_train, C_train, batch_size):
    dataset = ArrayDataset(X_train, C_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

def evaluate_model(model, data_samples, cond, cond_dim, condition, control, locs):
    device = data_samples.device
    model = model.to(device)

    # ground truth
    start_true = time.time()

    log_condition_true = -0.5 * ((data_samples.cpu().numpy() - np.array(locs[np.argmax(condition)])) ** 2).sum(axis=1) - 0.5 * data_samples.shape[1] * np.log(2 * np.pi)
    log_control_true = -0.5 * ((data_samples.cpu().numpy() - np.array(locs[np.argmax(control)])) ** 2).sum(axis=1) - 0.5 * data_samples.shape[1] * np.log(2 * np.pi)
    log_ratio_true = log_condition_true - log_control_true
    
    time_true = time.time() - start_true

    condition = torch.from_numpy(condition).float().to(device).expand(data_samples.shape[0], cond_dim)
    control = torch.from_numpy(control).float().to(device).expand(data_samples.shape[0], cond_dim)
    
    # naive evaluation
    start_hat = time.time()

    log_condition_hat = model.estimate_log_density(data_samples, condition, n_steps=100)
    log_control_hat = model.estimate_log_density(data_samples, control, n_steps=100)
    log_ratio_hat = log_condition_hat - log_control_hat
    
    time_hat = time.time() - start_hat

    # correction term - its own field
    start_hat_v2 = time.time()

    log_ratio_hat_v2 = model.estimate_log_density_ratio(data_samples, condition, control, cond, n_steps=100)

    time_hat_v2 = time.time() - start_hat_v2
    
    return [log_ratio_true, log_ratio_hat, log_ratio_hat_v2], [time_true, time_hat, time_hat_v2]

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    np.random.seed(42)
    os.makedirs("./runs/gaussian_tests/scripts/checkpoints/", exist_ok=True)

    n = cfg.num_dims
    N = 110_000
    cond_dim = 2
    batch_size = 512
    n_steps = 100_000
    n_tries = 3
    control = np.array([1, 0])
    condition = np.array([0, 1])
    locs = [[0 for _ in range(n)]] + [[2 for _ in range(n)]]
    sigma = 0
    sigma_min = 0

    names = ["true", "naive", "ct own rl"]

    if cfg.start_from_scratch:
        X_train, X_test, C_train, C_test = prepare_dataset(n, N, cond_dim, locs)

        results = {}

        latent_dim_last = 0
        time_feature_dim_last = 0
    else:
        try:
            with open(f"./runs/gaussian_tests/scripts/checkpoints/checkpoint_{cfg.num_dims}_{cfg.lr}_{cfg.cond_latent_dim}.pkl", "rb") as f:
                saved_vars = pickle.load(f)

            results = saved_vars["results"]
            latent_dim_last = saved_vars["latent_dim_last"]
            time_feature_dim_last = saved_vars["time_feature_dim_last"]
            X_train = saved_vars["X_train"]
            X_test = saved_vars["X_test"]
            C_train = saved_vars["C_train"]
            C_test = saved_vars["C_test"]
        except:
            print("Could not load checkpoint, starting from scratch")
            X_train, X_test, C_train, C_test = prepare_dataset(n, N, cond_dim, locs)

            results = {}

            latent_dim_last = 0
            time_feature_dim_last = 0

    lambda_t = lambda t: torch.sqrt((1 - (1 - sigma_min) * t) ** 2 + sigma * t * (1 - t))
    lambda_sp_t = lambda t: (sigma * (1 - 2 * t) - 2 * (1 - sigma_min) * (1 - (1 - sigma_min) * t)) / 2

    latent_dims = [32, 64, 128, 256, 512]
    time_feature_dims = [1, 32, 64, 128]

    for latent_dim in latent_dims:
        for time_feature_dim in time_feature_dims:
            if latent_dim < latent_dim_last:
                print(f"Skipping latent dim {latent_dim}, time feature dim {time_feature_dim} since already done")
                continue
            elif latent_dim == latent_dim_last and time_feature_dim <= time_feature_dim_last:
                print(f"Skipping latent dim {latent_dim}, time feature dim {time_feature_dim} since already done")
                continue
            print(f"Starting latent dim {latent_dim}, time feature dim {time_feature_dim}")


            tmp_results = {name: np.array([]) for name in names}
            time_results = {name: [] for name in names}
            for _ in tqdm(range(n_tries)):
                model = ConditionalFlowMatchingWithScore(
                    input_dim=n,
                    cond_dims=[cond_dim],
                    hidden_dims=[1024, 1024, 1024],
                    encoder_hidden_dims=[256],
                    encoder_out_dim=latent_dim,
                    lambda_t=lambda_t,
                    lambda_sp_t=lambda_sp_t,
                    betas=[0],
                    lr=cfg.lr,
                    time_feature_dim=time_feature_dim,
                    encoder_out_dim_cond=cfg.cond_latent_dim,
                )

                train_loader = build_train_loader(X_train, C_train, batch_size)
                trainer = L.Trainer(
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=1,
                    max_steps=n_steps,
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                )
                trainer.fit(model, train_dataloaders=train_loader)

                log_ratios, times = evaluate_model(model, X_test, C_test, cond_dim, condition, control, locs)

                for i, name in enumerate(names):
                    if tmp_results[name].shape[0] == 0:
                        tmp_results[name] = log_ratios[i].reshape(-1, 1)
                    else:
                        tmp_results[name] = np.concatenate([tmp_results[name], log_ratios[i].reshape(-1, 1)], axis=1)
                        
                    time_results[name].append(times[i])

            latent_dim_last = latent_dim
            time_feature_dim_last = time_feature_dim

            for name in names:
                if name == "true":
                    key =  "- - true"
                elif name == "naive":
                    key = str(latent_dim) + " " + str(time_feature_dim) + " naive"
                elif name == "ct own rl":
                    key = str(latent_dim) + " " + str(time_feature_dim) + " rl"
                else:
                    raise ValueError("Unknown value for name")
                
                if key in results:
                    results[key]["value"] = np.concatenate([results[key]["value"], tmp_results[name]], axis=1)
                    results[key]["time"] += time_results[name]
                else:
                    results[key] = {}
                    results[key]["value"] = tmp_results[name]
                    results[key]["time"] = time_results[name]

            with open(f"./runs/gaussian_tests/scripts/checkpoints/checkpoint_{cfg.num_dims}_{cfg.lr}_{cfg.cond_latent_dim}.pkl", "wb") as f:
                pickle.dump({
                    "results": results,
                    "latent_dim_last": latent_dim_last,
                    "time_feature_dim_last": time_feature_dim_last,
                    "X_train": X_train,
                    "X_test": X_test,
                    "C_train": C_train,
                    "C_test": C_test
                }, f)

    with open(f"./notebooks/gaussian_tests/sweep_results/sweep_loc2_{cfg.num_dims}_{cfg.lr}_{cfg.cond_latent_dim}.pkl", "wb") as f:
        pickle.dump(results, f)

    print("FINISHED")
    
if __name__ == "__main__":
    main()