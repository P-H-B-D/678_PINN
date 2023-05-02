from typing import Callable
import argparse

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import torchopt

from matplotlib.animation import FuncAnimation
from pinn import make_forward_fn

#Based off of https://github.com/madagra/basic-pinn. 

# Oscillator motion parameters
m = 1.0  # mass
k = 1.0  # spring constant
x0 = 1.0  # initial displacement
v0 = 0.0  # initial velocity

# Boundary conditions
X_BOUNDARY_1 = 0.0
F_BOUNDARY_1 = x0
X_BOUNDARY_2 = 0.0
F_BOUNDARY_2 = v0


def make_loss_fn(f: Callable, d2fdx2: Callable) -> Callable:
    def loss_fn(params: torch.Tensor, x: torch.Tensor):
        # interior loss
        f_value = f(x, params)
        interior = d2fdx2(x, params) + k / m * f_value

        # boundary losses
        x_boundary_1 = torch.tensor([X_BOUNDARY_1])
        f_boundary_1 = torch.tensor([F_BOUNDARY_1])

        x_boundary_2 = torch.tensor([X_BOUNDARY_2])
        f_boundary_2 = torch.tensor([F_BOUNDARY_2])

        boundary_1 = f(x_boundary_1, params) - f_boundary_1
        boundary_2 = d2fdx2(x_boundary_2, params) - f_boundary_2

        loss = nn.MSELoss()

        # Weighting of the loss
        weight_interior = 0.25 
        weight_boundary_1 = 0.37 
        weight_boundary_2 = 0.37 

        loss_value = (
            weight_interior * loss(interior, torch.zeros_like(interior))
            + weight_boundary_1 * loss(boundary_1, torch.zeros_like(boundary_1))
            + weight_boundary_2 * loss(boundary_2, torch.zeros_like(boundary_2))
        )

        return loss_value

    return loss_fn

def make_dynamic_loss_fn(f: Callable, d2fdx2: Callable) -> Callable:
    def loss_fn(params: torch.Tensor, x: torch.Tensor,epoch: int, num_epochs: int):
        # interior loss
        f_value = f(x, params)
        interior = d2fdx2(x, params) + k / m * f_value

        # boundary losses
        x_boundary_1 = torch.tensor([X_BOUNDARY_1])
        f_boundary_1 = torch.tensor([F_BOUNDARY_1])

        x_boundary_2 = torch.tensor([X_BOUNDARY_2])
        f_boundary_2 = torch.tensor([F_BOUNDARY_2])

        boundary_1 = f(x_boundary_1, params) - f_boundary_1
        boundary_2 = d2fdx2(x_boundary_2, params) - f_boundary_2

        loss = nn.MSELoss()

        # Weighting of the loss
        weight_interior = 0.25 + (1-(epoch/num_epochs))
        weight_boundary_1 = 0.37 + ((epoch/num_epochs))
        weight_boundary_2 = 0.37 + ((epoch/num_epochs))
        print(1-(epoch/num_epochs))

        loss_value = (
            weight_interior * loss(interior, torch.zeros_like(interior))
            + weight_boundary_1 * loss(boundary_1, torch.zeros_like(boundary_1))
            + weight_boundary_2 * loss(boundary_2, torch.zeros_like(boundary_2))
        )

        return loss_value

    return loss_fn


if __name__ == "__main__":

    # make it reproducible
    torch.manual_seed(2)

    # parse input from user
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num-hidden", type=int, default=4)
    parser.add_argument("-d", "--dim-hidden", type=int, default=20)
    parser.add_argument("-b", "--batch-size", type=int, default=100)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.8e-2)
    parser.add_argument("-e", "--num-epochs", type=int, default=500)

    args = parser.parse_args()

    # configuration
    num_hidden = args.num_hidden
    dim_hidden = args.dim_hidden
    batch_size = args.batch_size
    num_iter = args.num_epochs
    tolerance = 1e-8
    learning_rate = args.learning_rate
    domain = (-5.0, 5.0)

    # function versions of model forward, gradient and loss
    fmodel, params, funcs = make_forward_fn(
        num_hidden=num_hidden, dim_hidden=dim_hidden, derivative_order=2
    )

    f = funcs[0]
    d2fdx2 = funcs[2]
    loss_fn = make_dynamic_loss_fn(f, d2fdx2)
    og_loss_fn = make_loss_fn(f, d2fdx2)

    # choose optimizer with functional API using functorch
    optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=learning_rate))

    # train the model
    loss_evolution = []
    og_loss_evolution = []
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)
    x_eval_np = x_eval.detach().numpy()
    analytical_sol_fn = (
            lambda x: x0 * np.cos(np.sqrt(k / m) * x) + v0 / np.sqrt(k / m) * np.sin(np.sqrt(k / m) * x)
        )

    def update(i):
        global params
        ax1.clear()
        ax2.clear()
        ax3.clear()

        # Sample points in the domain randomly for each epoch
        x = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1])

        # Update the parameters
        og_loss = og_loss_fn(params, x)
        og_loss_evolution.append(og_loss.item())

        loss = loss_fn(params, x, i, num_iter)
        params = optimizer.step(loss, params)
        loss_evolution.append(loss.item())
        print("Epoch: {}, Loss: {}".format(i, loss.item()))

        x_sample_np = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1]).detach().numpy()
        f_eval = f(x_eval, params)

        # ax.scatter(x_sample_np, analytical_sol_fn(x_sample_np), color="red", label="Sample training points")
        ax1.plot(x_eval_np, f_eval.detach().numpy(), label="PINN solution at iter {}".format(i))
        ax1.plot(
            x_eval_np,
            analytical_sol_fn(x_eval_np),
            label=f"Analytic solution",
            color="green",
            alpha=0.75,
        )
        ax1.set(title="Simple Harmonic Oscillator PINN Solution\n"+r"$\frac{d^2f}{dt^2} = -\frac{k}{m}f(t),\ k=1,\ m=1,\ f(0)=1,\ f'(0)=0$", xlabel="t", ylabel="f(t)")
        ax1.set_ylim(-2,5)
        ax1.legend()

        # Dynamic Loss evolution plot
        ax2.plot(loss_evolution)
        ax2.set_title("Dynamic Loss evolution")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")

        # Static Loss evolution plot
        ax3.plot(og_loss_evolution)
        ax3.set_title("Static Loss evolution")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")

    anim = FuncAnimation(fig, update, frames=num_iter, interval=10, repeat=False)
    anim.save('dynamic_loss.gif', dpi=80, writer='imagemagick')
    plt.show()

    
    

