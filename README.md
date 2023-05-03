# 678_PINN

A Variety of Physics Informed Neural Network (PINN) demonstrations for PHYS678 (*Computing for Scientific Research, Yale Graduate School of Arts and Sciences*) Final Project, c/o Peter Bowman-Davis, 2023. Based off of the codebase of https://github.com/madagra/basic-pinn, I expand the framework to include another first order ODE (exponential curve) and two second order ODEs (simple harmonic oscilator and damped harmonic oscillator). Additionally, I explore the effects of hyperparameters on the performance of these systems in these simple demonstrations. Additionally, I implement a methodology for dynamically weighting the contribution of the interior and boundary term weights, decreasing them with iteration, resulting in performance enhancements for a variety of differential equations.

## Abstract

Physics-Informed-Neural-Networks (PINNs) offer an alternative numerical method to solving initial value problems (IVPs) in differential equations. Notably, they offer the competitive advantage of being highly parallelizable in comparison to conventional methods by nature of their main limiting factor being the chosen neural network hyperparameters. In this paper, I present a framework for setting up a generalizable method for solving IVPs using PINNs, discuss assorted drawbacks and propose possible solutions for future research. 


## Introduction

The concept of being "Physics-Informed" in the context of machine learning and neural networks is quite simple. It means that the differential equation(s) of interest are integrated into the loss function of a neural network-driven regression. This process involves incorporating certain aspects of the differential equation(s) into the neural network model, allowing the model to learn and predict based on these underlying physics principles.

There are several aspects of the differential equation(s) that are typically integrated into the loss function of a physics-informed neural network (PINN). These include the boundary value loss and the "residual" or "interior" loss. The boundary value loss is used to compare the neural network output with given initial value(s), while the residual loss is used to evaluate the autodiff of the neural network output using the given differential equation. By constructing the neural network loss function in such a way that it is minimized, the PINN can automatically satisfy the underlying differential equation(s).

The architecture of a basic estimator neural network for PINNs in PyTorch consists of N layers of M neurons each, with a hyperbolic tangent (Tanh) activation function. The architecture's hyperparameters include the number of hidden layers, the number of neurons per layer, the learning rate, the batch size, and the loss weighting. Careful selection of these hyperparameters is necessary, as they can significantly impact the network's learning rate.

One of the significant advantages of using a PINN over other numerical methods such as Runge-Kutta is that it is parallelizable and more efficient in high-dimensional or complex initial value problems (IVPs). Additionally, PINNs are uniquely flexible at integrating novel factors into the loss function, making them particularly useful for sparse or incomplete datasets. Furthermore, PINNs are composable with other numerical methods, allowing users to combine PINNs with other techniques to further improve their predictive capabilities.

Overall, Physics-Informed Neural Networks offer a powerful and flexible approach to solving differential equations in various fields, including physics, engineering, and other sciences. By incorporating the underlying physics principles directly into the neural network's loss function, the PINN can more accurately predict and learn from complex, high-dimensional data.

## Architecture

The generalized architecture of the neural network used in the proceeding experiments can be found in [pinn.py](https://github.com/P-H-B-D/678_PINN/blob/main/pinn.py). 

The loss function for a given differential equation may be constructed by finding an equation for the differential equation equal to zero, and using this as the interior loss function, e.g. from [exp.py](https://github.com/P-H-B-D/678_PINN/blob/main/exp.py): 
```
# interior loss
f_value = f(x, params)
interior = dfdx(x, params) - R * f_value #for exponential equation df/dt=Rf -> df/dt-Rf=0
```
Next, we define the loss function at the boundary condition by simply evaluating the differential equation at this point and finding the MSE versus zero: 
```
boundary = f(x_boundary, params) - f_boundary
```
Finally, we construct the final loss function b ased on the weights of the constituent losses:
```
loss = nn.MSELoss()
weight_interior = 8.0
weight_boundary = 1.0 
#total loss
loss_value = weight_interior * loss(interior, torch.zeros_like(interior)) + weight_boundary * loss(boundary, torch.zeros_like(boundary))
```


### Exponential: $\frac{df}{dt} = Rf(t),\ R\in\mathbb{R},\ f(0)=1$
![](https://github.com/P-H-B-D/678_PINN/blob/main/exponential.gif)

### Simple Harmonic Oscillator: $\frac{d^2f}{dt^2} = -\frac{k}{m}f(t),\ k=1,\ m=1,\ f(0)=1,\ f'(0)=0$
![](https://github.com/P-H-B-D/678_PINN/blob/main/Harmonic.gif)

### Damped Harmonic Oscillator: $\frac{d^2f}{dt^2} = -\frac{c}{m}\frac{df}{dt} - \frac{k}{m}f,\ c,k,m\in\mathbb{R},\ f(0)=1,\ \frac{df}{dt}(0)=0$
![](https://github.com/P-H-B-D/678_PINN/blob/main/dampedHarmonic.gif)

### Sparse Data Regression on $\frac{d^2f}{dt^2} = -\frac{k}{m}f(t),\ k=1,\ m=1,\ f(0)=1,\ f'(0)=0$
![](https://github.com/P-H-B-D/678_PINN/blob/main/HarmonicSparseData.gif)

### Dynamic Weighting of Interior and Boundary Terms
![](https://github.com/P-H-B-D/678_PINN/blob/main/dynamic_loss.gif)


References:
* NN Base Code https://github.com/madagra/basic-pinn 
* PINN Theory https://www.sciencedirect.com/science/article/pii/S0021999118307125
* PINN Theory https://www.nature.com/articles/s42254-021-00314-5  
* Dynamic Hyperparameters: https://proceedings.neurips.cc/paper_files/paper/2018/file/8051a3c40561002834e59d566b7430cf-Paper.pdf 
