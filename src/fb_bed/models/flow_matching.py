import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable
from torchdyn.core import NeuralODE
from .node_wrappers import NODEWrapper, NODEWrapper_with_ratio_tvf, NODEWrapper_with_trace_div, NODEWrapper_EIG


class ConditionEncoder(nn.Module):
    """Project one condition block into the shared condition space."""

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0):
        super().__init__()

        layers = []
        previous_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(previous_dim, dim),
                nn.SELU(),
                nn.Dropout(dropout),
            ])
            previous_dim = dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, condition):
        return self.encoder(condition)


class FlowMatchingMLP(nn.Module):
    """
    MLP model for flow matching with time conditioning.

    The model concatenates the input features with a scalar time value
    before passing them through a feedforward network.

    Args:
        input_dim (int, optional): Input feature dimension (including time if precomputed).
        hidden_dims (list, optional): List of hidden layer sizes.
        output_dim (int, optional): Output dimension.
        dropout (float, optional): Dropout probability.
    """

    def __init__(self, input_dim: int = 128, hidden_dims: list = [],
                 output_dim: int = 128, dropout: float = 0):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.SELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        """
        Forward pass with time concatenation.

        Args:
            x (torch.Tensor): Input tensor of shape (N, D).
            t (torch.Tensor): Time tensor (scalar or batch).

        Returns:
            torch.Tensor: Output of shape (N, output_dim).
        """
        if t.dim() == 0 or t.size()[0] == 1:
            t = t.expand(x.shape[0]).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        
        xt = torch.cat([x, t], dim=1)
        return self.mlp(xt)

class ConditionalFlowMatchingWithScore(L.LightningModule):
    """
    Conditional Flow Matching model with joint vector field and score learning.

    Args:
        y_dim (int): Dimension of the modeled variable y.
        x_dim (int): Dimension of condition x.
        theta_dim (int): Dimension of condition theta.
        eps_dim (int): Dimension of condition eps.
        hidden_dims (list): Hidden layer sizes for MLPs.
        encoder_hidden_dims (list): Hidden layer sizes for condition encoders.
        encoder_out_dim_cond (int): Shared condition embedding dimension.
        lambda_t (Callable): Lambda function.
        lambda_sp_t (Callable): Derivative of lambda function.
        betas (list): Dropout probabilities for x, theta, and eps.
        lr (float, optional): Learning rate.
        dropout (float, optional): Dropout probability.
    """

    def __init__(
        self,
        y_dim: int,
        x_dim: int,
        theta_dim: int,
        eps_dim: int,
        hidden_dims: list,
        encoder_hidden_dims: list,
        encoder_out_dim_cond: int,
        lambda_t: Callable,
        lambda_sp_t: Callable,
        betas: list,
        lr: float = 1e-4,
        dropout: float = 0
    ):
        super().__init__()

        self.cond_dims = [x_dim, theta_dim, eps_dim]
        if len(betas) != len(self.cond_dims):
            raise ValueError("betas must contain one value each for x, theta, and eps.")

        self.cond_encoders = nn.ModuleList([
            ConditionEncoder(
                condition_dim,
                encoder_hidden_dims,
                encoder_out_dim_cond,
                dropout,
            )
            for condition_dim in self.cond_dims
        ])

        model_input_dim = y_dim + encoder_out_dim_cond + 1
        self.vf_mlp = FlowMatchingMLP(
            model_input_dim,
            hidden_dims,
            y_dim,
            dropout
        )
        self.score_mlp = FlowMatchingMLP(
            model_input_dim,
            hidden_dims,
            y_dim,
            dropout
        )
        
        self.lambda_t = lambda_t
        self.lambda_sp_t = lambda_sp_t
        self.betas = betas
        self.y_dim = y_dim
        self.encoder_out_dim_cond = encoder_out_dim_cond
        self.lr = lr

    def forward(self, x, t, cond, use_conds=None):
        """
        Compute vector field and score.

        Args:
            x (torch.Tensor): Input data.
            t (torch.Tensor): Time tensor.
            cond (torch.Tensor): Concatenated condition tensor.
            use_conds (list, optional): Boolean mask for condition usage.

        Returns:
            tuple: (vector_field, score).
        """
        if use_conds is None:
            use_conds = torch.ones(
                x.shape[0], len(self.cond_dims), dtype=torch.bool, device=x.device
            )
        else:
            use_conds = torch.as_tensor(use_conds, dtype=torch.bool, device=x.device)
            if use_conds.ndim == 1:
                if use_conds.shape[0] != len(self.cond_dims):
                    raise ValueError(
                        "use_conds must contain one value each for x, theta, and eps."
                    )
                use_conds = use_conds.unsqueeze(0).expand(x.shape[0], -1)
            elif use_conds.shape != (x.shape[0], len(self.cond_dims)):
                raise ValueError(
                    "use_conds must have shape (3,) or (batch_size, 3)."
                )

        if t.dim() == 0 or t.size()[0] == 1:
            t = t.expand(x.shape[0]).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        condition_embeddings = []
        start = 0
        for condition_index, condition_dim in enumerate(self.cond_dims):
            condition_part = cond[:, start:start + condition_dim]
            condition_embedding = self.cond_encoders[condition_index](condition_part)
            condition_embedding = condition_embedding * use_conds[
                :, condition_index:condition_index + 1
            ]
            condition_embeddings.append(condition_embedding)
            start += condition_dim

        condition_embedding = torch.stack(condition_embeddings, dim=0).sum(dim=0)
        model_input = torch.cat([x, condition_embedding, t], dim=1)
        vf = self.vf_mlp.mlp(model_input)
        score = self.score_mlp.mlp(model_input)
        return vf, score

    def shared_step(self, x1, cond):
        """
        Compute training loss for a batch.

        Args:
            x1 (torch.Tensor): Target samples.
            cond (torch.Tensor): Conditioning tensor.

        Returns:
            torch.Tensor: Scalar loss.
        """
        device = x1.device

        x0 = torch.randn_like(x1).to(device)
        t = torch.rand(x1.shape[0]).unsqueeze(1).to(device)

        xt = t * x1 + self.lambda_t(t) * x0
        ut = x1 + self.lambda_sp_t(t) / self.lambda_t(t) * x0
        c_t = self.lambda_t(t) ** 2 - self.lambda_sp_t(t) * t

        use_conds = (
            torch.rand(x1.shape[0], len(self.betas), device=device)
            >= torch.as_tensor(self.betas, device=device)
        )
        pred_ut, pred_score = self(xt, t, cond, use_conds)
        
        vf_loss = F.mse_loss(pred_ut, ut)
        score_loss = F.mse_loss(c_t * pred_score, t * ut - xt)

        return vf_loss + score_loss

    def _unpack_batch(self, batch):
        """
        !!!CHANGE HERE ONCE DATASET IS UPDATED!!!

        Unpack batch into (x, cond).

        Args:
            batch: Tuple or list containing (x, cond).

        Returns:
            tuple: (x, cond).

        Raises:
            ValueError: If batch format is invalid.
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) != 2:
                raise ValueError("Expected batch to be (x, cond).")
            x, cond = batch
        else:
            raise ValueError("Expected batch to be (x, cond).")
        return x, cond

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.
        """
        x, cond = self._unpack_batch(batch)
        loss = self.shared_step(x, cond)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.
        """
        x, cond = self._unpack_batch(batch)
        loss = self.shared_step(x, cond)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch: Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss.
        """
        x, cond = self._unpack_batch(batch)
        loss = self.shared_step(x, cond)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizer.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def get_node(self, condition, control=None, point=None, use_conds=None, node_type="simulation", estimator_type="exact", solver="dopri"):
        """
        Construct Neural ODE wrapper.

        Args:
            condition (torch.Tensor): Condition tensor.
            control (torch.Tensor, optional): Control condition.
            point (torch.Tensor, optional): Point condition.
            use_conds (list, optional): List of boolean values for conditional inference.
            node_type (str): "simulation", "density", or "ratio".
            estimator_type (str): Divergence estimator type.
            solver (str): ODE solver type.

        Returns:
            NeuralODE: Configured Neural ODE instance.
        """
        if node_type == "simulation":
            return NeuralODE(NODEWrapper(self, condition), solver=solver,
                              sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        elif node_type == "density":
            return NeuralODE(NODEWrapper_with_trace_div(self, condition, estimator_type, use_conds), solver=solver,
                              sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        elif node_type == "ratio":
            return NeuralODE(NODEWrapper_with_ratio_tvf(self, condition, control, point, estimator_type), solver=solver,
                              sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        elif node_type == "eig":
            return NeuralODE(NODEWrapper_EIG(self, condition, estimator_type), solver=solver,
                              sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        else:
            raise ValueError(f"Unknown node_type: {node_type}. Must be 'simulation', 'density', 'ratio' or 'eig'.")
        
    def run_simulation(self, data_samples, condition, n_steps=100, solver="dopri5"):
        """
        Run forward ODE simulation.

        Args:
            data_samples (torch.Tensor): Input samples.
            condition (torch.Tensor): Condition tensor.
            n_steps (int): Number of integration steps.
            solver (str): ODE solver type.

        Returns:
            torch.Tensor: Simulated samples.
        """
        device = data_samples.device
        node = self.get_node(condition, node_type="simulation", solver=solver)
        
        with torch.no_grad():
            traj = node.trajectory(
                torch.randn_like(data_samples).to(device),
                t_span=torch.linspace(0, 1, n_steps).to(device)
            )
            
        return traj[-1].cpu().numpy()
        
    def estimate_log_density(self, data_samples, condition, n_steps=100, use_conds=None, estimator_type="hutch_gaussian", solver="dopri5"):
        """
        Estimate log-density via probability flow ODE.

        Args:
            data_samples (torch.Tensor): Input samples.
            condition (torch.Tensor): Condition tensor.
            n_steps (int): Integration steps.
            estimator_type (str): Divergence estimator.
            solver (str): ODE solver type.

        Returns:
            np.ndarray: Log-density estimates.
        """
        device = data_samples.device
        node = self.get_node(condition, node_type="density", estimator_type=estimator_type, solver=solver, use_conds=use_conds)
        
        with torch.no_grad():
            traj = node.trajectory(
                torch.cat([data_samples, torch.zeros(data_samples.shape[0], 1).to(device)], dim=-1),
                t_span=torch.linspace(1, 0, n_steps).to(device)
            )
        z0, div = traj[-1, :, :-1], traj[-1, :, -1]
        log_p1 = -0.5 * (z0 ** 2).sum(dim=1) - 0.5 * z0.shape[1] * np.log(2 * np.pi) + div
        
        return log_p1.cpu().numpy()
    
    def estimate_log_density_ratio(self, data_samples, condition, control, point, n_steps=100, estimator_type="hutch_gaussian", solver="dopri5"):
        """
        Estimate log-density ratio.

        Args:
            data_samples (torch.Tensor): Input samples.
            condition (torch.Tensor): Condition tensor.
            control (torch.Tensor): Control condition.
            point (torch.Tensor): Reference point condition.
            n_steps (int): Integration steps.
            estimator_type (str): Divergence estimator.
            solver (str): ODE solver type.

        Returns:
            np.ndarray: Log-density ratio estimates.
        """
        device = data_samples.device
        node = self.get_node(condition, control, point, node_type="ratio", estimator_type=estimator_type, solver=solver)
        
        with torch.no_grad():
            traj = node.trajectory(
                torch.cat([data_samples, torch.zeros(data_samples.shape[0], 1).to(device)], dim=-1),
                t_span=torch.linspace(1, 0, n_steps).to(device)
            )

        log_ratio = traj[-1, :, -1]
        return -log_ratio.cpu().numpy()

    def estimate_eig(self, data_samples, condition, n_steps=100, estimator_type="hutch_gaussian", solver="dopri5"):
        """
        Estimate EIG (Expected Information Gain).

        Args:
            data_samples (torch.Tensor): Input samples.
            condition (torch.Tensor): Condition tensor.
            n_steps (int): Integration steps.
            estimator_type (str): Divergence estimator.
            solver (str): ODE solver type.

        Returns:
            np.ndarray: EIG estimates.
        """
        device = data_samples.device
        node = self.get_node(condition, node_type="eig", estimator_type=estimator_type, solver=solver)
        
        traj = node.trajectory(
            torch.cat([data_samples, torch.zeros(data_samples.shape[0], 1).to(device)], dim=-1),
            t_span=torch.linspace(1, 0, n_steps).to(device)
        )
        
        return traj[-1, :, -1]