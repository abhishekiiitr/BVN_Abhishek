# Animate bivariate normal distribution
### Probelm Statement:
- Reproduce the above figure showing samples from bivariate normal with marginal PDFs from scratch using JAX and matplotlib.
- Add interactivity to the figure by adding sliders with ipywidgets. You should be able to vary the parameters of bivariate normal distribution (mean and covariance matrix) using ipywidgets.

## SOLUTION :
Install Jax library :
```sh
%pip install jax
```
Import modules of JAX and MATPLOTLIB
```sh
import jax
import jax.numpy as jnp
import numpy as np![bvn](https://user-images.githubusercontent.com/77018574/162633217-0d7ce56b-380c-466a-8297-4bbd5c12b67a.PNG)


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
```
NOW for slider to work import the slider module of ipywidgets
```sh
from ipywidgets import interact
```
##### JAX WORKING
- The X intermediate range is constructed with jax using the “arange” function.
- The Y intermediate range is constructed with jax using the “arange” function.
- The X, Y ranges are constructed with the “meshgrid” function from jax-numpy.
```sh
def jnp_bivariate_normal_pdf(domain, mean, variance):
  X = jnp.arange(-domain+mean, domain+mean, variance)
  Y = jnp.arange(-domain+mean, domain+mean, variance)
  X, Y = jnp.meshgrid(X, Y)
  R = jnp.sqrt(X**2 + Y**2)
  Z = ((1. / jnp.sqrt(2 * jnp.pi)) * jnp.exp(-.5*R**2))
  return X+mean, Y+mean, Z
```
##### Slider to interact
- here we used interact decorator to decorate the function, so the function can receive the slide bar's value with parameter x.
```sh
@interact(x=(0, 100))
def double_number(x):
    --pass
```
#### function of bivariable normal plotter
code snippet is
```sh
def plt_plot_bivariate_normal_pdf(x, y, z):
  fig = plt.figure(figsize=(12, 6))
  ax = fig.gca(projection='3d')
  ax.plot_surface(x, y, z, 
                  cmap=cm.coolwarm,
                  linewidth=0, 
                  antialiased=True)
  ax.set_xlabel('Random Variable x')
  ax.set_ylabel('Random variable y')
  ax.set_zlabel('BN PDF');
  plt.show()
```
 Now link the slider value to the plotter function for live interaction of 3D graph

```sh
@interact(a=(0,10),b=(0, 10),c=(0, 1.))
def domain_x(a,b,c):
  plt_plot_bivariate_normal_pdf(*jnp_bivariate_normal_pdf(a,b,c))
```
##### Output of the task is here-
 ![bvn](https://user-images.githubusercontent.com/77018574/162633253-bfa0edb5-7aae-4b69-bfc3-0f4b4faa90f8.PNG)
  
## References
- [Aly Shmahell](https://buymeacoff.ee/AlyShmahell) -towardsdatascience
- [dev2qa](https://www.dev2qa.com/how-to-add-interactive-widget-slide-bar-in-jupyter-notebook/)
- Stackoverflow



