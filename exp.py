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

R = 0.3  # rate of maximum population growth parameterizing the equation
X_BOUNDARY = 0.0  # boundary condition coordinate
F_BOUNDARY = 0.3  # boundary condition value


def make_loss_fn(f: Callable, dfdx: Callable) -> Callable:
    """Make a function loss evaluation function
    The loss is computed as sum of the interior MSE loss (the differential equation residual)
    and the MSE of the loss at the boundary
    Args:
        f (Callable): The functional forward pass of the model used a universal function approximator. This
            is a function with signature (x, params) where `x` is the input data and `params` the model
            parameters
        dfdx (Callable): The functional gradient calculation of the universal function approximator. This
            is a function with signature (x, params) where `x` is the input data and `params` the model
            parameters
    Returns:
        Callable: The loss function with signature (params, x) where `x` is the input data and `params` the model
            parameters. Notice that a simple call to `dloss = functorch.grad(loss_fn)` would give the gradient
            of the loss with respect to the model parameters needed by the optimizers
    """

    def loss_fn(params: torch.Tensor, x: torch.Tensor):

        # interior loss
        f_value = f(x, params)
        interior = dfdx(x, params) - R * f_value #for exponential equation df/dt=Rf -> df/dt-Rf=0

        # boundary loss
        x0 = X_BOUNDARY
        f0 = F_BOUNDARY
        x_boundary = torch.tensor([x0])
        f_boundary = torch.tensor([f0])
        boundary = f(x_boundary, params) - f_boundary
        loss = nn.MSELoss()
        weight_interior = 8.0
        weight_boundary = 1.0 
        #total loss
        loss_value = weight_interior * loss(interior, torch.zeros_like(interior)) + weight_boundary * loss(boundary, torch.zeros_like(boundary))
        
        # loss_value = loss(interior, torch.zeros_like(interior)) + loss(
        #     boundary, torch.zeros_like(boundary)
        # )

        return loss_value

    return loss_fn


if __name__ == "__main__":

    # make it reproducible
    torch.manual_seed(2)

    # parse input from user
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num-hidden", type=int, default=5)
    parser.add_argument("-d", "--dim-hidden", type=int, default=10)
    parser.add_argument("-b", "--batch-size", type=int, default=100)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-2)
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
        num_hidden=num_hidden, dim_hidden=dim_hidden, derivative_order=1
    )

    f = funcs[0]
    dfdx = funcs[1]
    loss_fn = make_loss_fn(f, dfdx)

    # choose optimizer with functional API using functorch
    optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=learning_rate))

    # train the model
    loss_evolution = []
    fig, ax = plt.subplots()
    x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)
    x_eval_np = x_eval.detach().numpy()
    analytical_sol_fn = lambda x: R*np.exp(R*x)

    def update(i):
        global params
        ax.clear()

        # Sample points in the domain randomly for each epoch
        x = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1])

        # Update the parameters
        loss = loss_fn(params, x)
        params = optimizer.step(loss, params)
        loss_evolution.append(loss.item())
        print("Epoch: {}, Loss: {}".format(i, loss.item()))

        x_sample_np = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1]).detach().numpy()
        f_eval = f(x_eval, params)

        # ax.scatter(x_sample_np, analytical_sol_fn(x_sample_np), color="red", label="Sample training points")
        ax.plot(x_eval_np, f_eval.detach().numpy(), label="PINN solution at iter {}".format(i))
        ax.plot(
            x_eval_np,
            analytical_sol_fn(x_eval_np),
            label=f"Analytic solution",
            color="green",
            alpha=0.75,
        )
        ax.set(title="Exponential PINN Solution\n"r"$\frac{df}{dt} = Rf(t),\ R\in\mathbb{R},\ f(0)=1$", xlabel="t", ylabel="f(t)")
        ax.set_ylim(-2,5)
        ax.legend()

    anim = FuncAnimation(fig, update, frames=num_iter, interval=10, repeat=False)
    #save to gif
    anim.save('exponential.gif', dpi=80, writer='imagemagick')

    plt.show()

    # plot loss evolution
    plt.plot(loss_evolution)
    plt.yscale('log')
    plt.title("Loss evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    
    

