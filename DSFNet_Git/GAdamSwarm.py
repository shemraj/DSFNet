import torch
import torch.optim as optim

class PSOAdamOptimizer(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameters: {}".format(betas))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(PSOAdamOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # Particle Swarm Optimization (PSO) update
                # The velocity will be updated based on the particle position and personal and global bests.
                if not hasattr(p, 'pso_velocity'):
                    p.pso_velocity = torch.zeros_like(p.data)
                    p.pso_best_position = p.data.clone()

                # Assuming the PSO hyperparameters (w, c1, and c2) are pre-defined.
                w, c1, c2 = 0.7, 2.0, 2.0
                r1, r2 = torch.rand_like(p.data), torch.rand_like(p.data)
                p.pso_velocity = w * p.pso_velocity + c1 * r1 * (p.pso_best_position - p.data) + c2 * r2 * (p.pso_best_position - p.data)
                p.data = p.data + p.pso_velocity

                # Adam update for backward propagation
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1
                beta1, beta2 = group['betas']
                state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * grad
                state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * grad ** 2

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

                p.data = p.data - step_size * state['exp_avg'] / (state['exp_avg_sq'] + group['eps']).sqrt()

        return loss
