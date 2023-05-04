# 678_PINN

A Variety of Physics Informed Neural Network (PINN) demonstrations for PHYS678 (*Computing for Scientific Research, Yale Graduate School of Arts and Sciences*) Final Project, c/o Peter Bowman-Davis, 2023. Based off of the codebase of https://github.com/madagra/basic-pinn, I expand the framework to include another first order ODE (exponential curve) and two second order ODEs (simple harmonic oscilator and damped harmonic oscillator). Additionally, I demonstrate a data-sparse implementation of these PINNs which allow for the learning process to be accelerated through sparse or rich sample data. Next, I explore the effects of varying hyperparameters on the performance of these systems in these simple demonstrations. Finally, I implement a novel methodology for dynamically weighting the contribution of the interior and boundary term weights, decreasing them with iteration, resulting in performance enhancements for a variety of differential equations.



## Abstract

Physics-Informed-Neural-Networks (PINNs) offer an alternative numerical method to solving initial value problems (IVPs) in differential equations. Notably, they offer the competitive advantage of being highly parallelizable in comparison to conventional methods by nature of their main limiting factor being the chosen neural network hyperparameters. In this paper, I present a framework for setting up a generalizable method for solving IVPs using PINNs, discuss assorted drawbacks and propose possible solutions for future research. 


## Introduction

The term, "Physics-Informed", in the context of machine learning and neural networks means that features of the differential equation(s) of interest are integrated into the loss function of a neural network-driven regression. This process involves incorporating certain aspects of the differential equation(s) into the neural network model, allowing the model to learn and predict based on these underlying physics principles.

There are several aspects of the differential equation(s) that are typically integrated into the loss function of a physics-informed neural network (PINN). These include the boundary value loss and the "residual" or "interior" loss. The boundary value loss is used to compare the neural network output with given initial value(s), while the residual loss is used to evaluate the autodiff of the neural network output using the given differential equation. By constructing the neural network loss function in such a way that it is minimized, the PINN can automatically satisfy the underlying differential equation(s). 

The architecture of a basic estimator neural network for PINNs can be implemented in PyTorch, consisting of N layers of M neurons each, with a hyperbolic tangent (Tanh) activation function. The architecture's hyperparameters include the number of hidden layers, the number of neurons per layer, the learning rate, the batch size, and the loss weighting. Careful selection of the loss weighting hyperparameters is necessary, as they can also affect the learning rate if not normalized.

One of the significant advantages of using a PINN over other numerical methods such as Runge-Kutta is that it is parallelizable and more efficient in high-dimensional or complex initial value problems (IVPs). Additionally, PINNs are uniquely flexible at integrating novel factors into the loss function, making them particularly useful for or incomplete datasets (as demonstrated in [simpleHarmonicExperimentalData.py](https://github.com/P-H-B-D/678_PINN/blob/main/simpleHarmonicExperimentalData.py)). Furthermore, PINNs are composable with other numerical methods, allowing users to combine PINNs with other techniques to further improve their predictive capabilities, notably by first computing sparse values for the function using RK or similar numerical methods, then integrating these outputs into the loss function. 

Physics-Informed Neural Networks offer a powerful and flexible approach to solving differential equations in various fields, including physics, engineering, and other sciences. By incorporating the underlying physics principles directly into the neural network's loss function, the PINN can more accurately predict and learn from complex, high-dimensional data. It should be noted that while the below implementations only solve ODEs of various order, it is possible to construct similar analogues for solving PDEs such as the heat equation or the Navier-Stokes Equation, which would use the partial derivatives in the loss function rather than the total derivatives. However, for the sake of simplicity, these examples will be constrained to ODEs.

## Architecture / Methodology

The generalized architecture of the neural network underpinning the proceeding experiments can be found in [pinn.py](https://github.com/P-H-B-D/678_PINN/blob/main/pinn.py). 

The loss function for a given differential equation may be constructed by finding an equation for the differential equation equal to zero, and using this as the interior loss function, e.g. from [exp.py](https://github.com/P-H-B-D/678_PINN/blob/main/exp.py): 
```python
# interior loss
f_value = f(x, params)
interior = dfdx(x, params) - R * f_value #for exponential equation df/dt=Rf -> df/dt-Rf=0
```
Next, we define the loss function at the boundary condition by simply evaluating the differential equation at this point and finding the MSE versus zero: 
```python
boundary = f(x_boundary, params) - f_boundary
```
Finally, we construct the final loss function based on the weights of the constituent losses:
```python
loss = nn.MSELoss()
weight_interior = 8.0
weight_boundary = 1.0 
#total loss
loss_value = weight_interior * loss(interior, torch.zeros_like(interior)) + weight_boundary * loss(boundary, torch.zeros_like(boundary))
```

This loss is sampled at various points along the domain (the amount of which samples is determined by the *batch-size* hyperparameter), and the adjustment is backpropogated using the standard adam optimizer. 

## Experiments
### Differential Equations Tested
#### Logistic [Logistic.py](https://github.com/P-H-B-D/678_PINN/blob/main/logistic.py):
![](https://github.com/P-H-B-D/678_PINN/blob/main/visuals/logistic.gif)

#### Exponential [exp.py](https://github.com/P-H-B-D/678_PINN/blob/main/exp.py): $\frac{df}{dt} = Rf(t),\ R\in\mathbb{R},\ f(0)=1$
![](https://github.com/P-H-B-D/678_PINN/blob/main/visuals/exponential.gif)

#### Simple Harmonic Oscillator [simpleHarmonic.py](https://github.com/P-H-B-D/678_PINN/blob/main/simpleHarmonic.py): $\frac{d^2f}{dt^2} = -\frac{k}{m}f(t),\ k=1,\ m=1,\ f(0)=1,\ f'(0)=0$
![](https://github.com/P-H-B-D/678_PINN/blob/main/visuals/Harmonic.gif)

#### Damped Harmonic Oscillator [dampedHarmonic.py](https://github.com/P-H-B-D/678_PINN/blob/main/dampedHarmonic.py): $\frac{d^2f}{dt^2} = -\frac{c}{m}\frac{df}{dt} - \frac{k}{m}f,\ c,k,m\in\mathbb{R},\ f(0)=1,\ \frac{df}{dt}(0)=0$
This is an example of a differential equation which the network struggles to converge to, even at hyperparameters that are well-suited to solving the problem. This is a common problem that arised throughout experimentation which highly irregular, complex, or oscillatory functions. In these cases, nonconvergence or partial convergence (in which some, but not all features would be converged to) was common. However, this could be alleviated by using a higher parameter model, modifying other hyperparameters such as learning rate, or increasing the num_epochs. 

![](https://github.com/P-H-B-D/678_PINN/blob/main/visuals/dampedHarmonic.gif)

#### Sparse Example Data [(simpleHarmonicExperimentalData.py)](https://github.com/P-H-B-D/678_PINN/blob/main/simpleHarmonicExperimentalData.py) on $\frac{d^2f}{dt^2} = -\frac{k}{m}f(t),\ k=1,\ m=1,\ f(0)=1,\ f'(0)=0$

It is very simple to incorporate data (sparse or rich) into PINNs by introducing an additional loss term, e.g. : 
```python
#data loss
f_data = f(x_data, params)
loss = nn.MSELoss()
data_loss = loss(f_data, y_data)
```
Which simply evaluates the loss of the PINN at the given datapoints in the current epoch. Note the performance benefits of this setup compared to the previous harmonic example, which uses identical hyperparameters.

![](https://github.com/P-H-B-D/678_PINN/blob/main/visuals/HarmonicSparseData.gif)


### Experimental Methodologies Tested

#### Weight Hyperparameter Variation  

The effects of varying weight hyperparameters can be elusive. The above demonstration shows the effects of weighting upon the resultant behavior of the system. While intuition as to the effects of each individual hyperparameter is relatively predictable, the combined effect of them on system stability or convergence is highly sensitive and unpredictable, making a hyperparameter search algorithm more ideal than simply guessing these values. A few permutations are shown here. Note that for each demonstration, the hyperparameter values have been normalized so that they do not affect the learning rate:

In the format (Interior, Boundary 1, Boundary 2), where Boundary 1 = f(0), Boundary 2= f'(0). Note how in the third example the network "snaps to" the initial value at f(0)=1, f'(0)=0, and in the preceeding two examples, the movement toward the IC is much more gradual: 

**(0.71,0.14,0.14)**:

![](https://github.com/P-H-B-D/678_PINN/blob/main/visuals/71.gif)

**(0.43,0.28,0.28)**:

![](https://github.com/P-H-B-D/678_PINN/blob/main/visuals/43.gif)

**(0.25,0.37,0.37)**:

![](https://github.com/P-H-B-D/678_PINN/blob/main/visuals/25.gif)

#### Dynamic Hyperparameter Weighting  

In a novel technique that I demonstrate in this presentation, I propose that certain systems may lend themselves to dynamically adjusting the weight hyperparameters based on epoch time. The basic idea behind this approach is to first ensure that the model is fitted to its initial value, and then adjust the weights towards interior losses. By dynamically altering the weight hyperparameters, the system can continuously improve and adapt to the changing loss function throughout the training process. This hypothetically allows for a more efficient and effective training of the model, ultimately resulting in better performance and accuracy. By dynamically changing the loss function, it becomes difficult for a human to assess model performance, since the criteria of assessment is changing over time. As a result, it is useful to construct a secondary "static" loss function for human evaluation purposes, which I plot alongside the dynamic loss (the loss that is fed into the optimizer for training). 

For example, in the construction of the loss function, one may add arguments for the epoch and max epoch into the loss function. The following code demonstrates a shift from loss weight being assigned to the interior toward loss weight being assigned to the boundary conditions:
```python
def loss_fn(params: torch.Tensor, x: torch.Tensor,epoch: int, num_epochs: int):
.
.
.
  weight_interior = (1-(epoch/num_epochs))
  weight_boundary_1 = ((epoch/num_epochs))
  weight_boundary_2 = ((epoch/num_epochs))
```

The code for this section may be found in [DynamicWeighting.py](https://github.com/P-H-B-D/678_PINN/blob/main/DynamicWeighting.py)

#### Interior -> Boundary: 
![](https://github.com/P-H-B-D/678_PINN/blob/main/visuals/dynamic_loss_interior_boundary.gif)

#### Boundary -> Interior: 
![](https://github.com/P-H-B-D/678_PINN/blob/main/visuals/dynamic_loss_boundary_interior.gif)

#### Baseline + Boundary -> Interior:

Noting that performance seems to be better on the boundary -> interior trial, we can try adding a baseline to the loss function, e.g. using 
```python
weight_interior = 0.35+((epoch/num_epochs))
weight_boundary_1 = 0.35+(1-(epoch/num_epochs))
weight_boundary_2 = 0.35+(1-(epoch/num_epochs))
```
This decreases the variance of the weight values, possibly providing a boost in performance over static methods (though this has yet to be evaluated). 
![](https://github.com/P-H-B-D/678_PINN/blob/main/visuals/baseline_dynamic_loss_interior_boundary.gif)

## Discussion and Summary

The use of Physics-Informed Neural Networks (PINNs) has gained significant popularity in recent years due to their ability to solve partial differential equations (PDEs) with high accuracy. However, one of the biggest challenges when using PINNs is finding the right hyperparameters. The process of selecting the optimal values for the hyperparameters can be extremely difficult and time-consuming. Fortunately, PINNs are parallelizable, which makes hyperparameter optimization easier with methods like Grid Search or Bayesian Optimization. 

Another challenge in using PINNs is finding hyperparameters when solving highly complex, high-frequency, or erratic functions. This complexity results in a smaller "convergence radius" for hyperparameters, making it even more challenging to get the right values. The weighting of boundary versus interior points is also highly nonlinear and can be difficult to understand intuitively. Furthermore, the architecture of the PDEs can be tricky to set up, which requires significant expertise in both physics and machine learning. Finally, it should be noted that the training methodology for PINNs is not conducive to out-of-bounds forecasting, which means that the network's accuracy may not be reliable when making predictions outside of the training data range. 

Despite these challenges, the simplicity of the PINN architecture paired with its scaleability makes it an attractive alternative to traditional differential equation solvers, particularly when numerical integration techniques are excessively computationally expensive. 

## References:
* NN Base Code: https://github.com/madagra/basic-pinn 
* PINN Theory: https://www.sciencedirect.com/science/article/pii/S0021999118307125
* PINN Theory: https://www.nature.com/articles/s42254-021-00314-5  
* PINN Theory: https://link.springer.com/article/10.1007/s10915-022-01939-z 
* PDE solving (Heat) with PINNs: https://asmedigitalcollection.asme.org/heattransfer/article/143/6/060801/1104439/Physics-Informed-Neural-Networks-for-Heat-Transfer
* Dynamic Hyperparameters: https://proceedings.neurips.cc/paper_files/paper/2018/file/8051a3c40561002834e59d566b7430cf-Paper.pdf 
