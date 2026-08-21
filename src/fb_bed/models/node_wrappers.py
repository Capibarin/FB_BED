import torch

class NODEWrapper(torch.nn.Module):
    """
    Wrapper adapting a model to ODE solver signature.

    Args:
        model (torch.nn.Module): Model returning a vector field.
        cond (torch.Tensor): Conditioning tensor.
        **kwargs: Extra arguments passed to model.
    """

    def __init__(self, model, cond, **kwargs):
        super().__init__()
        self.model = model
        self.cond = cond
        self.kwargs = kwargs

    def forward(self, t, x, *args, **kwargs):
        """
        Evaluate vector field at time t.

        Args:
            t (torch.Tensor or float): Time.
            x (torch.Tensor): State.
            *args: Unused.
            **kwargs: Unused.

        Returns:
            torch.Tensor: Vector field.
        """
        vf, _ = self.model(x, t, self.cond, **self.kwargs)
        return vf
    

def exact_div_fn(u):
    """
    Create exact divergence function via Jacobian trace.

    Args:
        u (Callable): Vector field function.

    Returns:
        Callable: Divergence function.
    """
    J = torch.func.jacrev(u)
    return lambda x, *args: torch.trace(J(x))


def div_fn_hutch_trace(u):
    """
    Create Hutchinson trace estimator for divergence.

    Args:
        u (Callable): Vector field function.

    Returns:
        Callable: Divergence estimator.
    """
    def div_fn(x, eps):
        _, vjpfunc = torch.func.vjp(u, x)
        return (vjpfunc(eps)[0] * eps).sum()

    return div_fn


def get_div_and_eps(likelihood_estimator):
    """
    Select divergence estimator and noise sampler.

    Args:
        likelihood_estimator (str): "exact", "hutch_gaussian", or "hutch_rademacher".

    Returns:
        tuple: (divergence_fn, eps_fn)

    Raises:
        NotImplementedError: If estimator is unknown.
    """
    if likelihood_estimator == "exact":
        return exact_div_fn, None
    if likelihood_estimator == "hutch_gaussian":
        return div_fn_hutch_trace, torch.randn_like
    if likelihood_estimator == "hutch_rademacher":

        def eps_fn(x):
            return torch.randint_like(x, low=0, high=2).float() * 2 - 1.0

        return div_fn_hutch_trace, eps_fn
    
    raise NotImplementedError(
        f"likelihood estimator {likelihood_estimator} is not implemented"
    )


class NODEWrapper_with_trace_div(torch.nn.Module):
    """
    NODE wrapper augmenting state with divergence.

    Args:
        model (torch.nn.Module): Model returning vector_field.
        condition (torch.Tensor): Conditioning tensor.
        likelihood_estimator (str, optional): Divergence estimator type.
    """

    def __init__(self, model, condition, likelihood_estimator="exact", use_conds=None):
        super().__init__()
        self.model = model
        self.cond = condition
        self.use_conds = use_conds
        self.div_fn, self.eps_fn = get_div_and_eps(likelihood_estimator)

    def forward(self, t, x, *args, **kwargs):
        """
        Compute augmented dynamics including divergence.

        Args:
            t (torch.Tensor or float): Time.
            x (torch.Tensor): Augmented state.
            *args: Unused.
            **kwargs: Unused.

        Returns:
            torch.Tensor: Concatenated [dx, div].
        """
        x = x[..., :-1]
        
        if self.eps_fn is None:
            div = torch.vmap(
                lambda state, condition: self.div_fn(
                    lambda y: self.model(
                        y.unsqueeze(0), t, condition.unsqueeze(0), use_conds=self.use_conds
                    )[0].squeeze()
                )(state)
            )(x, self.cond)
        else:
            div = torch.vmap(
                lambda state, condition, noise: self.div_fn(
                    lambda y: self.model(
                        y.unsqueeze(0), t, condition.unsqueeze(0), use_conds=self.use_conds
                    )[0].squeeze()
                )(state, noise)
            )(x, self.cond, self.eps_fn(x))

        dx, _ = self.model(x, t, self.cond, use_conds=self.use_conds)
        return torch.cat([dx, div[:, None]], dim=-1)
    

class NODEWrapper_with_ratio_tvf(torch.nn.Module):
    """
    NODE wrapper computing density ratio dynamics.

    Args:
        model (torch.nn.Module): Model returning (vector_field, score).
        condition (torch.Tensor): Condition for u.
        control (torch.Tensor): Condition for v.
        point (torch.Tensor): Condition for f.
        likelihood_estimator (str, optional): Divergence estimator type.
    """

    def __init__(self, model, condition, control, point, likelihood_estimator="exact"):
        super().__init__()
        self.model = model
        self.cond_v = control
        self.cond_u = condition
        self.cond_f = point
        self.div_fn, self.eps_fn = get_div_and_eps(likelihood_estimator)

    def forward(self, t, x, *args, **kwargs):
        """
        Compute augmented density ratio dynamics.

        Args:
            t (torch.Tensor or float): Time.
            x (torch.Tensor): Augmented state.
            *args: Unused.
            **kwargs: Unused.

        Returns:
            torch.Tensor: Concatenated [f_t, density_ratio_derivative].
        """
        x = x[..., :-1]
        
        if self.eps_fn is None:
            div = torch.vmap(
                lambda state, condition_u, condition_v: self.div_fn(
                    lambda y: (
                        self.model(y.unsqueeze(0), t, condition_v.unsqueeze(0))[0]
                        - self.model(y.unsqueeze(0), t, condition_u.unsqueeze(0))[0]
                    ).squeeze()
                )(state)
            )(x, self.cond_u, self.cond_v)
        else:
            div = torch.vmap(
                lambda state, condition_u, condition_v, noise: self.div_fn(
                    lambda y: (
                        self.model(y.unsqueeze(0), t, condition_v.unsqueeze(0))[0]
                        - self.model(y.unsqueeze(0), t, condition_u.unsqueeze(0))[0]
                    ).squeeze()
                )(state, noise)
            )(x, self.cond_u, self.cond_v, self.eps_fn(x))
        
        ut, score_u = self.model(x, t, self.cond_u)
        vt, score_v = self.model(x, t, self.cond_v)
        ft, _ = self.model(x, t, self.cond_f)  # Simulating field
        
        correction_term_u = torch.linalg.vecdot(ft - ut, score_u)
        correction_term_v = torch.linalg.vecdot(vt - ft, score_v)
        dr = div + correction_term_u + correction_term_v
        
        return torch.cat([ft, dr[:, None]], dim=-1)


class NODEWrapper_EIG(torch.nn.Module):
    """
    NODE wrapper computing EIG (Expected Information Gain).

    Args:
        model (torch.nn.Module): Model returning (vector_field, score).
        condition (torch.Tensor): Condition for u.
        control (torch.Tensor): Condition for v.
        point (torch.Tensor): Condition for f.
        likelihood_estimator (str, optional): Divergence estimator type.
    """

    def __init__(self, model, condition, likelihood_estimator="exact"):
        super().__init__()
        self.model = model
        self.cond_u = condition
        self.div_fn, self.eps_fn = get_div_and_eps(likelihood_estimator)

    def forward(self, t, x, *args, **kwargs):
        """
        Compute augmented EIG dynamics.

        Args:
            t (torch.Tensor or float): Time.
            x (torch.Tensor): Augmented state.
            *args: Unused.
            **kwargs: Unused.

        Returns:
            torch.Tensor: Concatenated [v_t, EIG_derivative].
        """
        x = x[..., :-1]
        
        if self.eps_fn is None:
            div = torch.vmap(
                lambda state, condition: self.div_fn(
                    lambda y: (
                        self.model(y.unsqueeze(0), t, condition.unsqueeze(0), use_conds=[True, False, True])[0]
                        - self.model(y.unsqueeze(0), t, condition.unsqueeze(0))[0] 
                    ).squeeze()
                )(state)
            )(x, self.cond_u)
        else:
            div = torch.vmap(
                lambda state, condition, noise: self.div_fn(
                    lambda y: (
                        self.model(y.unsqueeze(0), t, condition.unsqueeze(0), use_conds=[True, False, True])[0]
                        - self.model(y.unsqueeze(0), t, condition.unsqueeze(0))[0]
                    ).squeeze()
                )(state, noise)
            )(x, self.cond_u, self.eps_fn(x))
        
        ut, score_u = self.model(x, t, self.cond_u)
        vt, _ = self.model(x, t, self.cond_u, use_conds=[True, False, True])
        
        correction_term_u = torch.linalg.vecdot(vt - ut, score_u)
        dr = div + correction_term_u
        
        return torch.cat([vt, dr[:, None]], dim=-1)