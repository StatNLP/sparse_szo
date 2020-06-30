# -*- coding: utf-8 -*-

"""
Optimizers module: defines update strategies

__Note for step() function:__
need to take a shapshot of the current weights before you call closure(),
and reset the params after the closure()-call every time.

example:
```
_copy_w = to_params(self._params).clone() # deepcopy

assign_params(w + perturbation) # update params temporarily
feedback = closure()     # so as to call models.forward() on the changed weights
assign_params(_copy_w)   # then put back the temporal snapshot

evolve(update)
```

"""

import numpy as np

import torch
from torch.nn.utils.convert_parameters import parameters_to_vector as to_vector
from torch.nn.utils.convert_parameters import vector_to_parameters as to_params

from collections import defaultdict

import logging
logger = logging.getLogger(__name__)


class BaseEvolutionOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-1, mean=0.0, var=1e-1, mu=1e-1, beta=0.0,
                 max_grad_norm=0.0, prune_or_freeze='prune', init='last', num_samples=1):
        defaults = dict(lr=lr, mean=mean, variance=var)
        super(BaseEvolutionOptimizer, self).__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise ValueError("EvolutionOptimizer doesn't support per-parameter options "
                             "(parameter groups)")

        # model param alias (pointer to model.parameters(), list of torch.Parameter obj)
        self._params = self.param_groups[0]["params"]

        self.mu = mu # exploration rate, smoothing param
        self.beta = beta # momentum coefficient
        if self.beta > 0.0:
            self.momentum = torch.zeros_like(self.w) # velocity
        self.max_grad_norm = max_grad_norm

        self.noise = torch.distributions.normal.Normal(mean, var)

        self.exploration = [0.01, 0.02, 0.05, 0.1, 0.5]
        self.distances = [torch.distributions.uniform.Uniform(-e, e) for e in self.exploration]

        self.update_counter = 0
        self._mask = torch.ones(self._w_dim, dtype=torch.uint8)
        self.init_lr = lr
        self._num_samples = num_samples
        self.prune_or_freeze = prune_or_freeze
        #assert prune_or_freeze in ['none', 'prune', 'freeze']
        #   'none': no pruning
        # 'freeze': prune the perturbation only and keep prev weights
        #  'prune': prune both perturbations and weights
        self.init = init

        # reset values where mask = 0
        self.reset_vec = self.w if self.prune_or_freeze == 'freeze' else torch.zeros_like(self.w)

        self.g = torch.zeros_like(self.w)
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.dev_scores = None

    @property
    def w(self):
        """returns vector form of self._params
            __ATTENTION:__
            don't assign any value to self.w directly.
            (self.w changes automatically when self._params has been updated.)
        """
        return to_vector(self._params).clone().detach()

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, grad):
        self._g = grad

    @property
    def lr(self):
        return self.param_groups[0]["lr"]

    @lr.setter
    def lr(self, lr):
        self.param_groups[0]["lr"] = lr

    @property
    def _w_dim(self): # return int
        """weights dimension in vector form"""
        return list(self.w.shape)[0]

    def _update_param_by_vec(self, vec):
        """assign new parameter values"""
        #assert vec.shape == self._w_dim
        to_params(vec, self._params)

    def sample_perturbation(self, mask=True, normalize=False):
        """sample perturbation and mask it"""
        noise = self.noise.sample((self._w_dim,))
        #assert self._mask.shape == noise.shape
        if normalize:
            std, mean = torch.std_mean(noise, dim=0)
            noise = (noise - mean)/std
        if mask:
            noise = self.mu * self.apply_mask(noise)
        return noise

    def evolve(self, grad):
        """update weights"""
        # take a snapshot
        _copy_w = self.w

        # mask gradients
        masked_grad = self.apply_mask(grad)

        # gradient clipping
        if self.max_grad_norm > 0.0:
            clip_coef = self.max_grad_norm / (torch.norm(masked_grad, p=2) + 1e-6) # avoid null division
            if clip_coef < 1:
                masked_grad = torch.mul(masked_grad, clip_coef)

        # momentum
        if self.beta > 0.0:
            self.momentum = self.beta * self.momentum + (1-self.beta) * masked_grad
            masked_grad = self.momentum

        # update
        new_weights = _copy_w + self.lr * masked_grad

        self._update_param_by_vec(new_weights)
        self.update_counter += 1
        self.g = masked_grad

    def prune(self, survive_vec, num_to_prune, closure=None):
        """sparsify the mask"""
        cutoff = None
        # L1 pruning
        if closure is None:
            abs_w = torch.abs(survive_vec)
            sorted_vec, _ = torch.sort(abs_w)
            cutoff = sorted_vec[num_to_prune].item() # cutoff magnitude

            # update mask
            self._mask = torch.where(abs_w > cutoff, torch.ones_like(self._mask), torch.zeros_like(self._mask))

        # random mask selection
        elif (closure is not None) and callable(closure):
            best_mask, best_score = None, 0.0
            prev_score = closure() # numpy float # F(w, x) on test set
            scores = []
            for _ in range(50):
                candidate_mask = torch.ones_like(self._mask) # place holder with ones
                mask_idx = torch.randperm(self._w_dim) # permute indices to be masked
                candidate_mask[mask_idx[:num_to_prune]] = 0.0 # set zeros on the chosen indices
                assert candidate_mask.sum() == (self._w_dim - num_to_prune), \
                    (candidate_mask.sum(), self._w_dim - num_to_prune)
                _tmp_w = torch.where(candidate_mask.to(dtype=torch.bool), survive_vec, self.reset_vec)
                self._update_param_by_vec(_tmp_w)
                score = closure() # numpy float
                scores.append(score)
                if score > best_score:
                    best_score = score
                    best_mask = candidate_mask
                #self._update_param_by_vec(_copy_w)
            #assert max(scores) == best_score
            self._mask = best_mask
            self.dev_scores = (prev_score, best_score, min(scores), np.std(scores))

        elif closure == 'random':
            random_mask = torch.ones_like(self._mask) # place holder with ones
            mask_idx = torch.randperm(self._w_dim) # permute indices to be masked
            random_mask[mask_idx[:num_to_prune]] = 0.0
            self._mask =  random_mask
        else:
            raise ValueError

        # prune/freeze weights
        sparsified_vec = self.apply_mask(survive_vec, reset_vec=self.reset_vec)
        self._update_param_by_vec(sparsified_vec) # initial weights at the beginning of next round

        return cutoff

    def apply_mask(self, vec, reset_vec=None):
        """apply mask

        Parameters
        ----------
        vec : ``torch.tensor``, required.
            Vector with vales to survive in freezing. (mask = 1) random: randomly re-initialized, reset: initial w^0, best: w^*, last: w^t}
        reset_vec : ``torch.tensor``, optional.
            Vector with vales to be reset. (mask = 0) prune: zero vec, freeze: self.reset_vec
        """
        if reset_vec is None:
            reset_vec = torch.zeros_like(vec)
        return torch.where(self._mask.to(dtype=torch.bool), vec, reset_vec)

    def check_sparsity(self):
        """check sparsity of mask"""
        # check sparsity of mask
        num_active = self._mask.sum().to(dtype=torch.float32)
        return 1-(num_active / self._mask.numel()).item(), num_active.item()
    
    def initialize(self):
        """initialize optimizer"""
        self.lr = self.init_lr
        #self.update_counter = 0
        self.g = torch.zeros_like(self.w)
        
        # momentum
        if self.beta > 0.0 and hasattr(self, 'momentum'):
            self.momentum = torch.zeros_like(self.w)
            
        # control variates
        if hasattr(self, 'num_steps') and hasattr(self, 'avg_feedback'):
            self.num_steps = 0
            self.avg_feedback = 0.0

        if self.prune_or_freeze == 'freeze':
            if self.init == 'reset':
                # keep initial values throughout rounds
                pass
            elif self.init == 'last':
                # reset each round
                self.reset_vec = self.w
        elif self.prune_or_freeze == 'prune':
            # set to zero
            self.reset_vec = torch.zeros_like(self.w)

    def _gradients_to_vector(self): # first order true gradients
        """
        see:
        https://pytorch.org/docs/stable/_modules/torch/nn/utils/convert_parameters.html
        """
        grad_vec = []
        break_flag = False

        for param in self._params:
            if param.grad is None:
                break_flag = True
                break
            else:
                grad_vec.append(param.grad.view(-1).clone()) # deep copy

        if break_flag:
            return None
        return torch.cat(grad_vec)

    #def get_true_grad_norm(self):
    #    """compute L2-norm of true gradients"""
    #    #grad_norm = 0
    #    #for param in self._params:
    #    #    param_norm = param.grad.data.norm(2)
    #    #    grad_norm += param_norm.item() ** 2
    #    #grad_norm = grad_norm ** (1. / 2)
    #    #assert torch.abs(grad_norm - torch.norm(self._gradients_to_vector(), p=2)) < 1e-5
    #    return torch.norm(self._gradients_to_vector(), p=2)

    def get_approx_grad_var(self, closure, return_probs=False):
        assert hasattr(self, 'coeff') and hasattr(self, 'perturbations')
        #assert not torch.allclose(self.perturbations, torch.zeros((self._num_samples, self._w_dim)))
        diff_norm_list = torch.zeros(self._num_samples)

        true_grad = self.get_first_order_gradients(closure, return_probs)
        self.zero_grad() # just in case ...

        for i in range(self._num_samples):
            # || \nabla f(w + \mu u) - \nabla f(w) || ^2
            approx_grad = self.coeff[i] * self.perturbations[i]
            diff_norm = torch.norm(approx_grad - true_grad, p=2) ** 2
            diff_norm_list[i] = diff_norm

        return torch.var(diff_norm_list), torch.var(self.coeff)

    def get_first_order_gradients(self, closure, return_probs=False):
        """compute first-order gradients on a train minibatch"""
        self.zero_grad() # reset grad
        if return_probs:
            _, rewards = closure()
        else:
            rewards = closure() # map inference (loss)
        if self.n_gpu > 1:
            rewards = rewards.mean()
        rewards.backward() # compute gradients on a batch without accumulation!
        grad = self._gradients_to_vector() # get gradients
        return grad

    def estimate(self, closure):
        """compute estimated neighbourhood lipschitz on the whole test set"""
        neighbours = defaultdict(list)

        # take a snapshot
        _orig_w = self.w

        # true first order gradients \nabla F(w^t)
        self.zero_grad() # reset grad
        #assert torch.sum(self._gradients_to_vector() != 0.0) == 0.0, torch.sum(self._gradients_to_vector() != 0.0)
        num_accumulation = closure() # call loss.backward()
        true_grad = self._gradients_to_vector() # get vectorized gradients
        true_grad /= num_accumulation

        for e, uniform in zip(self.exploration, self.distances):
            for _ in range(self._num_samples):
                # get neighbour
                distance = uniform.sample((self._w_dim,))
                #assert self.w.shape == distance.shape
                self._update_param_by_vec(self.w + distance)

                # get true gradients
                # \nabla F(w^t + distance, x)
                self.zero_grad() # reset grad
                num_accumulation = closure() # call loss.backward()
                true_grad_neighbour = self._gradients_to_vector() # get vectorized gradients
                true_grad_neighbour /= num_accumulation

                # compute lipschitz
                # || \nabla F(w^t + distance, x) - \nabla F(w^t, x) || / || distance ||
                lipschitz = torch.norm(true_grad_neighbour - true_grad, p=2) / torch.norm(distance, p=2)
                neighbours[f'{e}'].append(lipschitz.item())

                # put the original w back
                self._update_param_by_vec(_orig_w)
                self.zero_grad() # just in case...

        return torch.norm(true_grad, p=2).detach().cpu().numpy(), neighbours

    def step(self, closure):
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure : ``callable``, required.
            A closure that re-evaluates the model and returns the function value (loss/reward).
        """
        raise NotImplementedError


class DuelingEvolutionOptimizer(BaseEvolutionOptimizer):
    def __init__(self, params, lr=1e-1, mean=0.0, var=1e-1, mu=1e-1, beta=0.0,
                 max_grad_norm=0.0, prune_or_freeze='prune', init='last', direction='descent', num_samples=0):
        super(DuelingEvolutionOptimizer, self).__init__(params, lr=lr, mean=mean, var=var, mu=mu, beta=beta,
            prune_or_freeze=prune_or_freeze, init=init, max_grad_norm=max_grad_norm, num_samples=num_samples)

        self.direction = direction

    def step(self, closure):
        # take a snapshot
        _orig_w = self.w

        # map inference
        map_feedback = closure() # = F(w, x)

        # noisy inference
        masked_perturbation = self.sample_perturbation(mask=True, normalize=False) # = \mu * \bar{u}
        self._update_param_by_vec(self.w + masked_perturbation)
        noisy_feedback = closure() # = F(w + \mu * \bar{u}, x)

        # reset params
        self._update_param_by_vec(_orig_w)

        # updates
        if self.direction == 'descent':
            if noisy_feedback > map_feedback:
                update = masked_perturbation
                self.evolve(update)
        elif self.direction == 'ascent':
            if noisy_feedback < map_feedback:
                update = -masked_perturbation
                self.evolve(update)

    def __repr__(self):
        return f"<DuelingEvolutionOptimizer(direction='{self.direction}')>"


class VanillaEvolutionOptimizer(BaseEvolutionOptimizer):
    def __init__(self, params, lr=1e-1, mean=0.0, var=1e-1, mu=1e-1, beta=0.0,
                 max_grad_norm=0.0, prune_or_freeze='prune', init='last', num_samples=1, cv=True):
        super(VanillaEvolutionOptimizer, self).__init__(params, lr=lr, mean=mean, var=var, mu=mu, beta=beta,
            prune_or_freeze=prune_or_freeze, init=init, max_grad_norm=max_grad_norm, num_samples=num_samples)

        self.perturbations = torch.zeros((self._num_samples, self._w_dim))
        self.coeff = torch.zeros(self._num_samples)

    def step(self, closure):
        # take a snapshot
        _orig_w = self.w


        self.perturbations = torch.zeros((self._num_samples, self._w_dim))
        self.coeff = torch.zeros(self._num_samples)

        approx_gradients = torch.zeros(self._w_dim)
        for i in range(self._num_samples):
            # noisy inference
            masked_perturbation = self.sample_perturbation(mask=True, normalize=False) # = \mu * \bar{u}
            self._update_param_by_vec(self.w + masked_perturbation)
            noisy_feedback = closure() # = F(w + \mu * \bar{u}, x)

            if self.n_gpu > 1:
                noisy_feedback = noisy_feedback.mean()

            # reset params
            self._update_param_by_vec(_orig_w)

            # g_mu(w) = F(w + \mu * \bar{u}, x) * \bar{u}
            approx_gradients += noisy_feedback * masked_perturbation

            # save
            self.perturbations[i] = masked_perturbation
            self.coeff[i] = noisy_feedback

        # average
        approx_gradients /= self._num_samples

        # update
        self.evolve(approx_gradients)

    def __repr__(self):
        return "<VanillaEvolutionOptimizer()>"


class OneSideEvolutionOptimizer(BaseEvolutionOptimizer):
    def __init__(self, params, lr=1e-1, mean=0.0, var=1e-1, mu=1e-1, beta=0.0,
                 max_grad_norm=0.0, prune_or_freeze='prune', init='last', num_samples=1):
        super(OneSideEvolutionOptimizer, self).__init__(params, lr=lr, mean=mean, var=var, mu=mu, beta=beta,
            prune_or_freeze=prune_or_freeze, init=init, max_grad_norm=max_grad_norm, num_samples=num_samples)

        self.perturbations = torch.zeros((self._num_samples, self._w_dim))
        self.coeff = torch.zeros(self._num_samples)

    def step(self, closure):
        # take a snapshot
        _orig_w = self.w

        # map inference
        map_feedback = closure()

        self.coeff = torch.zeros(self._num_samples)
        self.perturbations = torch.zeros((self._num_samples, self._w_dim))

        approx_gradients = torch.zeros(self._w_dim)
        for i in range(self._num_samples):
            # noisy inference
            masked_perturbation = self.sample_perturbation(mask=True, normalize=False) # = \mu * \bar{u}
            self._update_param_by_vec(self.w + masked_perturbation)
            noisy_feedback = closure() # = F(w + \mu * \bar{u}, x)

            if self.n_gpu > 1:
                map_feedback = map_feedback.mean()
                noisy_feedback = noisy_feedback.mean()

            # reset params
            self._update_param_by_vec(_orig_w)

            # g_mu(w) = F(w + \mu * \bar{u}, x) / \mu * \bar{u}
            coeff = (noisy_feedback - map_feedback) / self.mu # \R (scalar)
            approx_gradients += coeff * masked_perturbation # \R^n (vector in length n)

            # save
            self.perturbations[i] = masked_perturbation
            self.coeff[i] = coeff

        # average
        approx_gradients /= self._num_samples

        # update
        self.evolve(approx_gradients)

    def __repr__(self):
        return "<OneSideEvolutionOptimizer()>"


class TwoSideEvolutionOptimizer(BaseEvolutionOptimizer):
    def __init__(self, params, lr=1e-1, mean=0.0, var=1e-1, mu=1e-1, beta=0.0,
                 max_grad_norm=0.0, prune_or_freeze='prune', init='last', num_samples=1):
        super(TwoSideEvolutionOptimizer, self).__init__(params, lr=lr, mean=mean, var=var, mu=mu, beta=beta,
            prune_or_freeze=prune_or_freeze, init=init, max_grad_norm=max_grad_norm, num_samples=num_samples)

        self.perturbations = torch.zeros((self._num_samples, self._w_dim))
        self.coeff = torch.zeros(self._num_samples)

    def step(self, closure):
        # take a snapshot
        _orig_w = self.w

        self.coeff = torch.zeros(self._num_samples)
        self.perturbations = torch.zeros((self._num_samples, self._w_dim))

        approx_gradients = torch.zeros(self._w_dim)
        for i in range(self._num_samples):

            masked_perturbation = self.sample_perturbation(mask=True, normalize=False) # = \mu * \bar{u}
            self._update_param_by_vec(self.w + masked_perturbation)
            noisy_feedback_pos = closure()     # = F(w + \mu * \bar{u}, x)

            self._update_param_by_vec(_orig_w) # reset to orig w

            self._update_param_by_vec(self.w - masked_perturbation)
            noisy_feedback_neg = closure()     # = F(w - \mu * \bar{u}, x)

            self._update_param_by_vec(_orig_w) # reset to orig w

            if self.n_gpu > 1:
                noisy_feedback_pos = noisy_feedback_pos.mean()
                noisy_feedback_neg = noisy_feedback_neg.mean()

            coeff = (noisy_feedback_pos - noisy_feedback_neg) / (2 * self.mu)
            approx_gradients += coeff * masked_perturbation

            # save
            self.perturbations[i] = masked_perturbation
            self.coeff[i] = coeff

        # average
        approx_gradients /= self._num_samples

        # update
        self.evolve(approx_gradients)

    def __repr__(self):
        return "<TwoSideEvolutionOptimizer()>"


class FirstOrderOptimizer(BaseEvolutionOptimizer):
    def __init__(self, params, lr=1e-1, mean=0.0, var=1e-1, mu=1e-1, beta=0.0,
                 max_grad_norm=0.0, prune_or_freeze='prune', init='last', num_samples=0):
        super(FirstOrderOptimizer, self).__init__(params, lr=lr, mean=mean, var=var, mu=mu, beta=beta,
            prune_or_freeze=prune_or_freeze, init=init, max_grad_norm=max_grad_norm, num_samples=num_samples)

    def step(self, closure):
        self.zero_grad()

        # compute loss and its gradient
        map_feedback = closure()
        if self.n_gpu > 1:
            map_feedback = map_feedback.mean()
        map_feedback.backward()

        grad = self._gradients_to_vector()

        # SGD update
        self.evolve(grad)

    def __repr__(self):
        return "<FirstOrderOptimizer()>"


class FirstOrderBanditOptimizer(FirstOrderOptimizer):
    def __init__(self, params, lr=1e-1, mean=0.0, var=1e-1, mu=1e-1, beta=0.0, max_grad_norm=0.0,
                 prune_or_freeze='prune', init='last', cv=False, num_samples=0):
        super(FirstOrderBanditOptimizer, self).__init__(params, lr=lr, mean=mean, var=var, mu=mu, beta=beta,
            prune_or_freeze=prune_or_freeze, init=init, max_grad_norm=max_grad_norm, num_samples=num_samples)

        # control variates
        self.cv = cv
        if self.cv:
            self.num_steps = 0
            self.avg_feedback = 0.0

    def step(self, closure):
        self.zero_grad()

        bandit_feedback, rewards = closure()
        if self.n_gpu > 1:
            bandit_feedback = bandit_feedback.mean()
            rewards = rewards.mean()
        rewards.backward()

        grad = self._gradients_to_vector()

        if self.cv:
            self.num_steps += 1
            self.avg_feedback += (bandit_feedback - self.avg_feedback) / self.num_steps
            bandit_feedback -= self.avg_feedback

        # First Order Bandit update
        update = bandit_feedback * grad
        self.evolve(update)

    def __repr__(self):
        return f"<FirstOrderBanditOptimizer(cv={self.cv})>"
