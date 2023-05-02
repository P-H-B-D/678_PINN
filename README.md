# 678_PINN

![](https://github.com/P-H-B-D/678_PINN/blob/main/logistic.gif)

A Variety of Physics Informed Neural Network (PINN) demonstrations for PHYS678 (*Computing for Scientific Research, Yale Graduate School of Arts and Sciences*) Final Project, c/o Peter Bowman-Davis, 2023.

Based off of the codebase of https://github.com/madagra/basic-pinn, I expand the framework to include another first order ODE (exponential curve) and two second order ODEs (simple harmonic oscilator and damped harmonic oscillator). Additionally, I explore the effects of hyperparameters on the performance of these systems in these simple demonstrations. 

(IN PROGRESS): Additionally, I implement a methodology for dynamically weighting the contribution of the interior and boundary term weights, decreasing them with iteration, resulting in performance enhancements for a variety of differential equations.

### Exponential
![](https://github.com/P-H-B-D/678_PINN/blob/main/exponential.gif)

### Simple Harmonic Oscillator
![](https://github.com/P-H-B-D/678_PINN/blob/main/Harmonic.gif)

### Damped Harmonic Oscillator
![](https://github.com/P-H-B-D/678_PINN/blob/main/dampedHarmonic.gif)

### Sparse Data Regression
![](https://github.com/P-H-B-D/678_PINN/blob/main/HarmonicSparseData.gif)


References:
* NN Base Code https://github.com/madagra/basic-pinn 
* PINN Theory https://www.sciencedirect.com/science/article/pii/S0021999118307125
* PINN Theory https://www.nature.com/articles/s42254-021-00314-5  
* Dynamic Hyperparameters: https://proceedings.neurips.cc/paper_files/paper/2018/file/8051a3c40561002834e59d566b7430cf-Paper.pdf 
