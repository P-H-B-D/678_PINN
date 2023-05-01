# 678_PINN

![](https://github.com/P-H-B-D/678_PINN/blob/main/logistic.gif)

A Variety of Physics Informed Neural Network (PINN) demonstrations for PHYS678 (*Computing for Scientific Research, Yale Graduate School of Arts and Sciences*) Final Project, c/o Peter Bowman-Davis, 2023.

Based off of the codebase of https://github.com/madagra/basic-pinn, I expand the framework to include another first order ODE (exponential curve), second order ODEs (oscillatory motion), and 2-D PDEs (heat diffusion on a plate).

Additionally, I implement a methodology for dynamically weighting the contribution of the interior and boundary term weights, decreasing them with iteration, resulting in performance enhancements for a variety of differential equations.

### Exponential
![](https://github.com/P-H-B-D/678_PINN/blob/main/exponential.gif)

### Simple Harmonic Oscillator
![](https://github.com/P-H-B-D/678_PINN/blob/main/Harmonic.gif)

### Damped Harmonic Oscillator
![](https://github.com/P-H-B-D/678_PINN/blob/main/dampedHarmonic.gif)


References:
* https://github.com/madagra/basic-pinn 
* https://www.sciencedirect.com/science/article/pii/S0021999118307125
