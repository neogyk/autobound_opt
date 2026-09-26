"""SafeRate: PyTorch optimizers with a monotone-decrease-safe step size.

This is a PyTorch translation of the "SafeRate" optimizer from Chapter 5 of
the AutoBound paper (https://arxiv.org/abs/2212.11429), whose reference
implementation (JAX) lives in
autobound/autobound/notebooks/safe_learning_rates.ipynb.

Algorithm (unchanged from the notebook):
  1. At iterate x_t, take an update direction v_t (for plain `SafeRate` this
     is -grad f(x_t); see `SafeAdamW` below for a preconditioned variant).
  2. Build a quadratic model  h(eta) = f(x_t) + c1*eta + c2*eta**2  of
     f(x_t + eta*v_t), valid over a trust region eta in [0, max_eta_t], with
     c1 = h'(0) computed exactly and c2 chosen so that h upper-bounds the true
     loss on the trust region. Note c1 = grad f(x_t) . v_t regardless of how
     v_t was chosen, so this step doesn't care whether v_t is the raw
     gradient or some preconditioned direction.
  3. Minimize the quadratic model over the trust region in closed form to get
     a step size eta_t that is guaranteed (given a valid c2) to not increase
     the loss: f(x_t + eta_t*v_t) <= f(x_t).
  4. Double max_eta if the step landed in the interior of the trust region
     (a sign the region was too conservative), halve it if the step landed
     at the trust-region boundary. If v_t wasn't actually a descent
     direction for this closure (c1 >= 0, so eta_t = 0 regardless of
     max_eta -- possible for `SafeAdamW`, see below), leave max_eta
     unchanged, since that step provides no evidence either way.

One deviation from AutoBound, unavoidable outside of JAX: AutoBound derives
c2 with certified interval Taylor arithmetic that traces the whole
computation graph, which guarantees the upper bound for *any* function.
PyTorch has no equivalent interval-arithmetic autograd mode, so both
optimizers below instead estimate c2 = 0.5 * max(h''(eta)) by evaluating the
exact directional second derivative d^T H(x_t + eta*v_t) d (via a
Hessian-vector product, i.e. double backprop) at a handful of points
spanning the trust region. This is an empirical rather than certified bound:
it is a reliable majorizer for functions whose curvature along v_t doesn't
spike between sample points, but -- unlike the JAX version -- it is not a
mathematical guarantee for arbitrary functions. Increase `curvature_samples`
for a more thorough (still empirical) search.
"""

import math
from typing import Callable, List, Optional, Tuple

import torch
from torch.optim import Optimizer


def minimize_1d_quadratic(
    c1: torch.Tensor, c2: torch.Tensor,
    trust_region: Tuple[float, float]) -> torch.Tensor:
  """Minimizes c1*x + c2*x**2 over trust_region.

  Direct translation of the `minimize_1d_quadratic` helper defined in
  autobound/autobound/notebooks/safe_learning_rates.ipynb.
  """
  a, b = trust_region
  a = torch.as_tensor(a, dtype=c1.dtype, device=c1.device)
  b = torch.as_tensor(b, dtype=c1.dtype, device=c1.device)

  def project(x):
    return torch.minimum(b, torch.maximum(a, x))

  def q(x):
    return c1 * x + c2 * x**2

  # If c2 > 0, set the derivative to zero to get -c1/(2*c2), then project
  # this into the trust region to get the minimizer.
  c2_pos_solution = project(-c1 / (2 * c2))
  # If c2 < 0, the minimizer is one of the two endpoints.
  c2_neg_solution = torch.where(q(a) < q(b), a, b)
  # If c2 == 0, the minimizer depends on the sign of c1.
  c2_zero_solution = torch.where(
      c1 == 0, project(torch.zeros_like(c1)), torch.where(c1 > 0, a, b))
  c2_nonzero_solution = torch.where(c2 > 0, c2_pos_solution, c2_neg_solution)
  return torch.where(c2 == 0, c2_zero_solution, c2_nonzero_solution)


class _SafeLineSearchOptimizer(Optimizer):
  """Shared machinery for "pick a direction, then take a SafeRate step".

  Subclasses provide the update direction (see `_compute_direction`); this
  base class handles building the quadratic majorizer along that direction,
  solving for the step size, applying it, and adapting the trust region.
  Unlike most `torch.optim` optimizers, there is no `lr` to tune: the step
  size is derived at every iteration from the local shape of the loss.

  Requires a closure, evaluated possibly several times per `step()` call,
  that recomputes and returns the loss *without* calling `.backward()` --
  these optimizers run autograd themselves (including double backward for
  curvature estimation).
  """

  def __init__(self, params, initial_max_eta: float, curvature_samples: int,
               defaults: dict):
    if initial_max_eta <= 0:
      raise ValueError(f'initial_max_eta must be positive, got {initial_max_eta}')
    if curvature_samples < 1:
      raise ValueError(f'curvature_samples must be >= 1, got {curvature_samples}')
    super().__init__(params, defaults)
    if len(self.param_groups) != 1:
      raise ValueError(
          f'{type(self).__name__} performs a single global line search per '
          "step and so doesn't support per-parameter options (multiple "
          'param groups).')
    self._params: List[torch.nn.Parameter] = [
        p for p in self.param_groups[0]['params'] if p.requires_grad
    ]
    if not self._params:
      raise ValueError(f'{type(self).__name__} got no parameters that '
                        'require grad.')
    # The trust region size is global, not per-parameter. Stash it on the
    # first parameter, mirroring torch.optim.LBFGS. Fetched fresh (via
    # `_state`) rather than cached, since load_state_dict() replaces
    # `self.state` wholesale.
    global_state = self.state[self._params[0]]
    global_state.setdefault('max_eta', float(initial_max_eta))
    global_state.setdefault('step', 0)
    global_state.setdefault('last_eta', 0.0)

  @property
  def _state(self) -> dict:
    return self.state[self._params[0]]

  def _compute_direction(
      self, grads: List[torch.Tensor]) -> List[torch.Tensor]:
    """Returns the (detached) update direction v_t given grad f(x_t)."""
    raise NotImplementedError

  @torch.no_grad()
  def _shift_params(self, direction: List[torch.Tensor], alpha: float) -> None:
    for p, d in zip(self._params, direction):
      p.add_(d, alpha=alpha)

  def _directional_curvature(self, closure: Callable[[], torch.Tensor],
                              direction: List[torch.Tensor],
                              eta: float) -> torch.Tensor:
    """Returns d^T H(x + eta*d) d, i.e. h''(eta) for h(t) = f(x + t*d)."""
    if eta != 0.0:
      self._shift_params(direction, eta)
    try:
      with torch.enable_grad():
        loss = closure()
        grads = torch.autograd.grad(loss, self._params, create_graph=True)
        hvp = torch.autograd.grad(grads, self._params, grad_outputs=direction)
    finally:
      if eta != 0.0:
        self._shift_params(direction, -eta)
    return sum(torch.sum(h * d) for h, d in zip(hvp, direction))

  def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
    """Performs a single step.

    Args:
      closure: a callable that recomputes and returns the loss at the
        current parameter values. Must NOT call `loss.backward()`.

    Returns:
      The loss evaluated at the parameters *before* this step.
    """
    if closure is None:
      raise ValueError(f'{type(self).__name__} requires a closure that '
                        'reevaluates the loss (without calling backward()).')
    group = self.param_groups[0]
    max_eta = self._state['max_eta']

    with torch.enable_grad():
      loss = closure()
      grads = list(
          torch.autograd.grad(loss, self._params, create_graph=True))
    direction = self._compute_direction(grads)
    # c1 = h'(0) = grad f(x) . direction, whatever `direction` is.
    c1 = sum(torch.sum(g.detach() * d) for g, d in zip(grads, direction))

    # Sample h''(eta) at eta=0 (reusing the graph built above) plus
    # `curvature_samples - 1` more points spanning (0, max_eta].
    hvp0 = torch.autograd.grad(grads, self._params, grad_outputs=direction)
    curvature_samples = [sum(torch.sum(h * d) for h, d in zip(hvp0, direction))]
    n_extra = group['curvature_samples'] - 1
    for i in range(n_extra):
      eta = max_eta if n_extra == 1 else max_eta * (i + 1) / n_extra
      curvature_samples.append(
          self._directional_curvature(closure, direction, eta))

    # c2 is the coefficient of eta**2 in the quadratic majorizer, i.e.
    # 0.5 * (empirical upper bound on h'' over the trust region).
    c2 = 0.5 * torch.stack(curvature_samples).max()

    safe_eta = minimize_1d_quadratic(c1, c2, (0.0, max_eta))
    safe_eta_val = float(safe_eta)

    self._shift_params(direction, safe_eta_val)

    if float(c1) >= 0:
      # `direction` wasn't a descent direction for this closure to begin
      # with (h'(0) >= 0), so minimize_1d_quadratic correctly returned
      # eta=0 -- but that has nothing to do with whether max_eta was well
      # sized. This happens for `SafeRate` only when the gradient is
      # exactly zero (converged); for direction choices that aren't
      # guaranteed descent directions for every closure call, such as
      # `SafeAdamW`'s momentum-based direction disagreeing with a noisy
      # mini-batch gradient, it can happen often. Leave the trust region
      # alone rather than shrinking it on a step that told us nothing about
      # its size.
      pass
    elif not math.isfinite(safe_eta_val) or safe_eta_val < max_eta / 2:
      self._state['max_eta'] = max_eta / 2
    else:
      self._state['max_eta'] = max_eta * 2
    self._state['last_eta'] = safe_eta_val
    self._state['step'] += 1

    return loss.detach()


class SafeRate(_SafeLineSearchOptimizer):
  """Steepest descent with a quadratic-majorizer, monotone-decrease step size.

  Example::

      optimizer = SafeRate(model.parameters())
      def closure():
          optimizer.zero_grad()  # optional; SafeRate doesn't rely on .grad
          return loss_fn(model(x), y)
      loss = optimizer.step(closure)

  Args:
    params: iterable of parameters to optimize (a single param group).
    initial_max_eta: initial size of the trust region, i.e. the largest step
      size considered on the first iteration.
    curvature_samples: number of points in [0, max_eta] (including both
      endpoints when >= 2) at which the directional curvature d^T H d is
      evaluated to build the empirical upper bound on h''. Must be >= 1.
  """

  def __init__(self,
               params,
               initial_max_eta: float = 1.0,
               curvature_samples: int = 2):
    defaults = dict(
        initial_max_eta=initial_max_eta, curvature_samples=curvature_samples)
    super().__init__(params, initial_max_eta, curvature_samples, defaults)

  def _compute_direction(self, grads):
    return [-g.detach() for g in grads]


class SafeAdamW(_SafeLineSearchOptimizer):
  """AdamW's preconditioned direction, taken with a SafeRate step size.

  This combines the two optimizers rather than just running them
  side-by-side: it reuses AdamW's exponential moving averages of the
  gradient and its square (and AdamW's decoupled weight decay) to build the
  same update *direction* AdamW would take, but instead of scaling that
  direction by a fixed (or scheduled) `lr`, it picks the step size the
  SafeRate way -- by majorizing the loss along that direction with a
  quadratic and minimizing it over an adaptive trust region. There is no
  `lr` hyperparameter as a result; `betas`/`eps`/`weight_decay` retain their
  usual AdamW meaning.

  The c1/c2 majorizer math in `_SafeLineSearchOptimizer.step` doesn't assume
  the direction is the negative gradient -- c1 = grad f(x) . v_t holds for
  any v_t -- so plugging in AdamW's preconditioned direction is a valid use
  of the same machinery, subject to the same empirical-bound caveat as
  `SafeRate` (see module docstring).

  Example::

      optimizer = SafeAdamW(model.parameters(), weight_decay=0.01)
      def closure():
          return loss_fn(model(x), y)
      loss = optimizer.step(closure)
  """

  def __init__(self,
               params,
               initial_max_eta: float = 1.0,
               curvature_samples: int = 2,
               betas: Tuple[float, float] = (0.9, 0.999),
               eps: float = 1e-8,
               weight_decay: float = 0.01):
    defaults = dict(
        initial_max_eta=initial_max_eta,
        curvature_samples=curvature_samples,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay)
    super().__init__(params, initial_max_eta, curvature_samples, defaults)

  def _compute_direction(self, grads):
    group = self.param_groups[0]
    beta1, beta2 = group['betas']
    eps = group['eps']
    weight_decay = group['weight_decay']
    # All parameters share one step count (they're all updated together
    # every call), taken one step ahead for this update's bias correction.
    step = self._state['step'] + 1
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    direction = []
    for p, g in zip(self._params, grads):
      g = g.detach()
      state = self.state[p]
      exp_avg = state.setdefault('exp_avg', torch.zeros_like(p))
      exp_avg_sq = state.setdefault('exp_avg_sq', torch.zeros_like(p))
      exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
      exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
      m_hat = exp_avg / bias_correction1
      v_hat = exp_avg_sq / bias_correction2
      # Decoupled weight decay, as in AdamW; scaled by eta at apply time
      # just as AdamW scales it by lr.
      direction.append(-(m_hat / (v_hat.sqrt() + eps) + weight_decay * p.detach()))
    return direction


if __name__ == '__main__':
  # Reproduces the 1-d quartic example from
  # autobound/autobound/notebooks/safe_learning_rates.ipynb, as a smoke test.
  x = torch.nn.Parameter(torch.tensor(0.0))
  optimizer = SafeRate([x], initial_max_eta=1.0)

  def closure():
    return (x - 3)**4

  for step in range(100):
    cur_loss = closure()
    optimizer.step(closure)
    if step & (step - 1) == 0:  # step is 0 or a power of two.
      state = optimizer._state
      print(f'step={step} loss={cur_loss.item():.6g} '
            f'eta={state["last_eta"]:.6g} max_eta={state["max_eta"]:.6g}')
