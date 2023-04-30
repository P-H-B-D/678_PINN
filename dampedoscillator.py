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
x0 = 2.0  # initial displacement
v0 = 0.0  # initial velocity
c = 0.2  # damping coefficient

# Boundary conditions
X_BOUNDARY_1 = 0.0
F_BOUNDARY_1 = x0
X_BOUNDARY_2 = 0.0
F_BOUNDARY_2 = v0


def make_loss_fn(f: Callable, dfdx: Callable, d2fdx2: Callable) -> Callable:
    def loss_fn(params: torch.Tensor, x: torch.Tensor):
        # interior loss
        f_value = f(x, params)
        dfdx_value = dfdx(x, params)
        interior = d2fdx2(x, params) + c / m * dfdx_value + k / m * f_value


        # boundary losses
        x_boundary_1 = torch.tensor([X_BOUNDARY_1])
        f_boundary_1 = torch.tensor([F_BOUNDARY_1])

        x_boundary_2 = torch.tensor([X_BOUNDARY_2])
        f_boundary_2 = torch.tensor([F_BOUNDARY_2])

        boundary_1 = f(x_boundary_1, params) - f_boundary_1
        boundary_2 = d2fdx2(x_boundary_2, params) - f_boundary_2

        loss = nn.MSELoss()

        # Weighting of the loss
        weight_interior = 20.0
        weight_boundary_1 = 2.0
        weight_boundary_2 = 2.0

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
    parser.add_argument("-lr", "--learning-rate", type=float, default=1.1e-2)
    parser.add_argument("-e", "--num-epochs", type=int, default=1000)

    args = parser.parse_args()

    # configuration
    num_hidden = args.num_hidden
    dim_hidden = args.dim_hidden
    batch_size = args.batch_size
    num_iter = args.num_epochs
    tolerance = 1e-8
    learning_rate = args.learning_rate
    domain = (0, 10.0)

    # function versions of model forward, gradient and loss
    fmodel, params, funcs = make_forward_fn(
        num_hidden=num_hidden, dim_hidden=dim_hidden, derivative_order=2
    )

    f = funcs[0]
    dfdx = funcs[1]
    d2fdx2 = funcs[2]
    loss_fn = make_loss_fn(f, dfdx, d2fdx2)

    # choose optimizer with functional API using functorch
    optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=learning_rate))

    # train the model
    loss_evolution = []
    fig, ax = plt.subplots()
    x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)
    x_eval_np = x_eval.detach().numpy()
    omega = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(m * k))
    omega_damped = omega * np.sqrt(1 - zeta**2)

    analytical_sol_fn = (
        lambda x: x0 * np.exp(-zeta * omega * x) * (np.cos(omega_damped * x) + zeta / np.sqrt(1 - zeta**2) * np.sin(omega_damped * x))
    )

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
        ax.set(title="Damped Harmonic Oscillator PINN Solution\n"+r"$\frac{d^2f}{dt^2} = -\frac{c}{m}\frac{df}{dt} - \frac{k}{m}f,\ c,k,m\in\mathbb{R},\ f(0)=1,\ \frac{df}{dt}(0)=0$", xlabel="t", ylabel="f(t)")
        ax.set_ylim(-2,5)
        ax.legend()

    anim = FuncAnimation(fig, update, frames=num_iter, interval=10, repeat=False)
    #save to gif
    anim.save('dampedHarmonic.gif', dpi=80, writer='imagemagick')

    plt.show()

    # plot loss evolution
    plt.plot(loss_evolution)
    plt.title("Loss evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    
    # Forecasting and plot
    extended_domain = (domain[0], domain[1] + 10)
    x_extended_eval = torch.linspace(extended_domain[0], extended_domain[1], steps=200).reshape(-1, 1)
    x_extended_eval_np = x_extended_eval.detach().numpy()

    f_extended_eval = f(x_extended_eval, params)

    f_step_eval = []
    prev_value = x_eval[-1]  # Last value from the training domain

    for i in range(100):  # 100 steps for forecasting
        x_step = prev_value + 0.1  # Assuming 0.1 increments in x
        f_value = f(torch.tensor([[x_step]]), params)  # Evaluate the model
        f_step_eval.append(f_value.item())
        prev_value = x_step

    f_step_eval = np.array(f_step_eval)

    # Combine training and forecasting results
    x_combined = np.concatenate((x_eval_np, np.linspace(domain[1] + 0.1, extended_domain[1], 100).reshape(-1, 1)))
    f_combined = np.concatenate((f(x_eval, params).detach().numpy(), f_step_eval.reshape(-1, 1)))

    # Plot the step-by-step forecast
    plt.plot(x_combined, f_combined, label="Step-by-step Forecast", color="purple")
    plt.plot(x_extended_eval_np, analytical_sol_fn(x_extended_eval_np), label="Analytical Solution", color="blue")

    # Highlight the forecasted portion
    plt.axvspan(domain[1], extended_domain[1], alpha=0.1, color="red")

    plt.title("Step-by-step PINN Forecast vs Analytical Solution")
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.legend()
    plt.show()

    

