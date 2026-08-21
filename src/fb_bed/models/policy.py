import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianExperimentalDesignPolicy(L.LightningModule):
	"""Recurrent policy for selecting the next experiment.

	Each history step is represented by the concatenation of ``y``, ``x``,
	and the experiment ``eps`` that produced that observation. The policy
	returns the next continuous experiment parameters.

	Args:
		y_dim: Dimension of each observation y.
		x_dim: Dimension of each context x.
		eps_dim: Dimension of each experiment eps.
		hidden_dim: GRU hidden-state dimension.
		hidden_dims: Hidden layer sizes in the output head.
		lr: Learning rate used by ``configure_optimizers``.
		dropout: Dropout probability in the output head.
		eps_min: Optional lower bound for the output.
		eps_max: Optional upper bound for the output.
	"""

	def __init__(
		self,
		y_dim: int,
		x_dim: int,
		eps_dim: int,
		hidden_dim: int = 128,
		hidden_dims: list[int] | None = None,
		lr: float = 1e-4,
		dropout: float = 0.0,
		eps_min: torch.Tensor | float | None = None,
		eps_max: torch.Tensor | float | None = None,
	):
		super().__init__()
		hidden_dims = hidden_dims or [hidden_dim]

		self.y_dim = y_dim
		self.x_dim = x_dim
		self.eps_dim = eps_dim
		self.lr = lr
		self.eps_min = eps_min
		self.eps_max = eps_max

		self.history_encoder = nn.GRU(
			input_size=y_dim + x_dim + eps_dim,
			hidden_size=hidden_dim,
			batch_first=True,
		)

		layers = []
		previous_dim = hidden_dim
		for dim in hidden_dims:
			layers.extend([
				nn.Linear(previous_dim, dim),
				nn.SELU(),
				nn.Dropout(dropout),
			])
			previous_dim = dim
		layers.append(nn.Linear(previous_dim, eps_dim))
		self.policy_head = nn.Sequential(*layers)

	def forward(self, y_history, x_history, eps_history, history_mask=None):
		"""Predict the next experiment from a batched experiment history.

		Args:
			y_history: Tensor with shape ``(batch, history, y_dim)``.
			x_history: Tensor with shape ``(batch, history, x_dim)``.
			eps_history: Tensor with shape ``(batch, history, eps_dim)``.
			history_mask: Optional boolean tensor with shape ``(batch, history)``.
				True marks a valid history step; False marks padding.

		Returns:
			Tensor with shape ``(batch, eps_dim)`` containing the next experiment.
		"""
		if y_history.ndim != 3 or x_history.ndim != 3 or eps_history.ndim != 3:
			raise ValueError("Histories must have shape (batch, history, features).")
		if y_history.shape[:2] != x_history.shape[:2] or y_history.shape[:2] != eps_history.shape[:2]:
			raise ValueError("y_history, x_history, and eps_history must share batch and history dimensions.")

		history = torch.cat([y_history, x_history, eps_history], dim=-1)
		encoded_history, _ = self.history_encoder(history)

		if history_mask is None:
			summary = encoded_history[:, -1]
		else:
			if history_mask.shape != history.shape[:2]:
				raise ValueError("history_mask must have shape (batch, history).")
			valid_steps = history_mask.to(device=history.device, dtype=torch.long).sum(dim=1)
			if torch.any(valid_steps == 0):
				raise ValueError("Each history must contain at least one valid step.")
			summary = encoded_history[
				torch.arange(history.shape[0], device=history.device), valid_steps - 1
			]

		next_eps = self.policy_head(summary)
		if self.eps_min is not None or self.eps_max is not None:
			next_eps = torch.clamp(next_eps, min=self.eps_min, max=self.eps_max)
		return next_eps

	def shared_step(self, batch):
		"""Compute supervised MSE loss for a target next experiment."""
		if len(batch) == 4:
			y_history, x_history, eps_history, next_eps = batch
			history_mask = None
		elif len(batch) == 5:
			y_history, x_history, eps_history, history_mask, next_eps = batch
		else:
			raise ValueError(
				"Expected (y_history, x_history, eps_history, next_eps) "
				"or (y_history, x_history, eps_history, history_mask, next_eps)."
			)

		prediction = self(y_history, x_history, eps_history, history_mask)
		return F.mse_loss(prediction, next_eps)

	def training_step(self, batch, batch_idx):
		loss = self.shared_step(batch)
		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		loss = self.shared_step(batch)
		self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
		return loss

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.lr)
