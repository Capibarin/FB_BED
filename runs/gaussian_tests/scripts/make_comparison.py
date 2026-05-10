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

    X_test = torch.tensor(X_test).to("cuda").float()
    C_test = F.one_hot(torch.tensor(C_test).long(), num_classes=cond_dim).to("cuda").float()
    return X_train, X_val, X_test, C_train, C_val, C_test

def build_train_loader(x_train, c_train, batch_size):
    dataset = ArrayDataset(x_train, c_train)
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

def get_params(num_dims):
    if num_dims == 2:
        time_feature_dim = 32
        latent_dim = 32
        cond_latent_dim = 64
    elif num_dims == 5:
        time_feature_dim = 1
        latent_dim = 256
        cond_latent_dim = 64
    elif num_dims == 10:
        time_feature_dim = 32
        latent_dim = 32
        cond_latent_dim = 32
    elif num_dims == 20:
        time_feature_dim = 1
        latent_dim = 64
        cond_latent_dim = 128
    elif num_dims == 30:
        time_feature_dim = 1
        latent_dim = 32
        cond_latent_dim = 128
    elif num_dims == 50:
        time_feature_dim = 32
        latent_dim = 256
        cond_latent_dim = 32
    else:
        raise ValueError("Unknown num_dims value")
        
    return latent_dim, time_feature_dim, cond_latent_dim

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    np.random.seed(42)
    os.makedirs("./runs/gaussian_tests/scripts/checkpoints/", exist_ok=True)

    n = cfg.num_dims
    N = cfg.num_samples
    cond_dim = 2
    batch_size = 512
    n_steps = 100_000
    n_tries = 3
    control = np.array([1, 0])
    condition = np.array([0, 1])
    locs = [[0 for _ in range(n)]] + [[cfg.loc for _ in range(n)]]
    sigma = 0
    sigma_mins = [1e-1, 1e-2, 1e-3, 1e-4, 0]

    latent_dim, time_feature_dim, cond_latent_dim = get_params(cfg.num_dims)

    names = ["true", "naive", "ct own rl"] 

    if cfg.start_from_scratch:
        X_train, X_val, X_test, C_train, C_val, C_test = prepare_dataset(n, N, cond_dim, locs)

        val_results = {}
        test_results = {}

        sigma_min_last = 2
    else:
        try:
            with open(f"./runs/gaussian_tests/scripts/checkpoints/main_checkpoint_{cfg.loc}_{cfg.num_dims}_{cfg.num_samples}.pkl", "rb") as f:
                saved_vars = pickle.load(f)

            val_results = saved_vars["val_results"]
            test_results = saved_vars["test_results"]
            sigma_last = saved_vars["sigma_last"]
            sigma_min_last = saved_vars["sigma_min_last"]
            X_train = saved_vars["X_train"]
            X_val = saved_vars["X_val"]
            X_test = saved_vars["X_test"]
            C_train = saved_vars["C_train"]
            C_val = saved_vars["C_val"]
            C_test = saved_vars["C_test"]
        except:
            print("Could not load checkpoint, starting from scratch")
            X_train, X_val, X_test, C_train, C_val, C_test = prepare_dataset(n, N, cond_dim, locs)

            val_results = {}
            test_results = {}

            sigma_min_last = 2

    for sigma_min in sigma_mins:
        if sigma_min >= sigma_min_last:
            print(f"Skipping sigma={sigma}, sigma_min={sigma_min} since already done")
            continue
        else:
            print(f"Starting sigma={sigma}, sigma_min={sigma_min}")

        lambda_t = lambda t: torch.sqrt((1 - (1 - sigma_min) * t) ** 2 + sigma * t * (1 - t))
        lambda_sp_t = lambda t: (sigma * (1 - 2 * t) - 2 * (1 - sigma_min) * (1 - (1 - sigma_min) * t)) / 2
        
        val_tmp_results = {name: np.array([]) for name in names}
        test_tmp_results = {name: np.array([]) for name in names}
        val_time_results = {name: [] for name in names}
        test_time_results = {name: [] for name in names}
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
                lr=1e-4,
                time_feature_dim=time_feature_dim,
                encoder_out_dim_cond=cond_latent_dim,
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

            val_log_ratios, val_times = evaluate_model(model, X_val, C_val, cond_dim, condition, control, locs)
            test_log_ratios, test_times = evaluate_model(model, X_test, C_test, cond_dim, condition, control, locs)
    
            for i, name in enumerate(names):
                if val_tmp_results[name].shape[0] == 0:
                    val_tmp_results[name] = val_log_ratios[i].reshape(-1, 1)
                else:
                    val_tmp_results[name] = np.concatenate([val_tmp_results[name], val_log_ratios[i].reshape(-1, 1)], axis=1)

                val_time_results[name].append(val_times[i])

                if test_tmp_results[name].shape[0] == 0:
                    test_tmp_results[name] = test_log_ratios[i].reshape(-1, 1)
                else:
                    test_tmp_results[name] = np.concatenate([test_tmp_results[name], test_log_ratios[i].reshape(-1, 1)], axis=1)

                test_time_results[name].append(test_times[i])

        for name in names:
            if name == "true":
                key =  "- - true"
            elif name == "naive":
                key = str(sigma_min) + " " + str(sigma) + " naive"
            elif name == "ct own rl":
                key = str(sigma_min) + " " + str(sigma) + " rl"
            else:
                raise ValueError("Unknown value for name")
            
            if key in val_results:
                val_results[key]["value"] = np.concatenate([val_results[key]["value"], val_tmp_results[name]], axis=1)
                val_results[key]["time"] += val_time_results[name]
            else:
                val_results[key] = {}
                val_results[key]["value"] = val_tmp_results[name]
                val_results[key]["time"] = val_time_results[name]

            if key in test_results:
                test_results[key]["value"] = np.concatenate([test_results[key]["value"], test_tmp_results[name]], axis=1)
                test_results[key]["time"] += test_time_results[name]
            else:
                test_results[key] = {}
                test_results[key]["value"] = test_tmp_results[name]
                test_results[key]["time"] = test_time_results[name]

        sigma_last = sigma
        sigma_min_last = sigma_min

        with open(f"./runs/gaussian_tests/scripts/checkpoints/main_checkpoint_{cfg.loc}_{cfg.num_dims}_{cfg.num_samples}.pkl", "wb") as f:
            pickle.dump({
                "val_results": val_results,
                "test_results": test_results,
                "sigma_last": sigma_last,
                "sigma_min_last": sigma_min_last,
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "C_train": C_train,
                "C_val": C_val,
                "C_test": C_test
            }, f)

    sigmas = [1, 0.75, 0.5, 0.25]
    sigma_min = 0

    for sigma in sigmas:
        print(f"Starting sigma={sigma}, sigma_min={sigma_min}")
        
        lambda_t = lambda t: torch.sqrt((1 - (1 - sigma_min) * t) ** 2 + sigma * t * (1 - t))
        lambda_sp_t = lambda t: (sigma * (1 - 2 * t) - 2 * (1 - sigma_min) * (1 - (1 - sigma_min) * t)) / 2
        
        val_tmp_results = {name: np.array([]) for name in names}
        test_tmp_results = {name: np.array([]) for name in names}
        val_time_results = {name: [] for name in names}
        test_time_results = {name: [] for name in names}
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
                lr=1e-4,
                time_feature_dim=time_feature_dim,
                encoder_out_dim_cond=cond_latent_dim,
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

            val_log_ratios, val_times = evaluate_model(model, X_val, C_val, cond_dim, condition, control, locs)
            test_log_ratios, test_times = evaluate_model(model, X_test, C_test, cond_dim, condition, control, locs)

            for i, name in enumerate(names):
                if val_tmp_results[name].shape[0] == 0:
                    val_tmp_results[name] = val_log_ratios[i].reshape(-1, 1)
                else:
                    val_tmp_results[name] = np.concatenate([val_tmp_results[name], val_log_ratios[i].reshape(-1, 1)], axis=1)

                val_time_results[name].append(val_times[i])

                if test_tmp_results[name].shape[0] == 0:
                    test_tmp_results[name] = test_log_ratios[i].reshape(-1, 1)
                else:
                    test_tmp_results[name] = np.concatenate([test_tmp_results[name], test_log_ratios[i].reshape(-1, 1)], axis=1)

                test_time_results[name].append(test_times[i])

        for name in names:
            if name == "true":
                key =  "- - true"
            elif name == "naive":
                key = str(sigma_min) + " " + str(sigma) + " naive"
            elif name == "ct own rl":
                key = str(sigma_min) + " " + str(sigma) + " rl"
            else:
                raise ValueError("Unknown value for name")
            
            if key in val_results:
                val_results[key]["value"] = np.concatenate([val_results[key]["value"], val_tmp_results[name]], axis=1)
                val_results[key]["time"] += val_time_results[name]
            else:
                val_results[key] = {}
                val_results[key]["value"] = val_tmp_results[name]
                val_results[key]["time"] = val_time_results[name]

            if key in test_results:
                test_results[key]["value"] = np.concatenate([test_results[key]["value"], test_tmp_results[name]], axis=1)
                test_results[key]["time"] += test_time_results[name]
            else:
                test_results[key] = {}
                test_results[key]["value"] = test_tmp_results[name]
                test_results[key]["time"] = test_time_results[name]

    mask_condition = np.all(C_test.cpu().numpy() == condition, axis=1)
    mask_control = np.all(C_test.cpu().numpy() == control, axis=1)
    mask_both = mask_condition | mask_control 

    val_results["mask_condition"] = mask_condition
    val_results["mask_control"] = mask_control
    val_results["mask_both"] = mask_both

    test_results["mask_condition"] = mask_condition
    test_results["mask_control"] = mask_control
    test_results["mask_both"] = mask_both

    with open(f"./notebooks/gaussian_tests/table_results/val_results_icml_{cfg.loc}_{cfg.num_dims}_{cfg.num_samples}.pkl", "wb") as f:
        pickle.dump(val_results, f)

    with open(f"./notebooks/gaussian_tests/table_results/test_results_icml_{cfg.loc}_{cfg.num_dims}_{cfg.num_samples}.pkl", "wb") as f:
        pickle.dump(test_results, f)

    print("FINISHED")
    
if __name__ == "__main__":
    main()