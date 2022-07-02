%pip install jax
import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from ipywidgets import interact

def jnp_bivariate_normal_pdf(domain, mean, variance):
  X = jnp.arange(-domain+mean, domain+mean, variance)
  Y = jnp.arange(-domain+mean, domain+mean, variance)
  X, Y = jnp.meshgrid(X, Y)
  R = jnp.sqrt(X**2 + Y**2)
  Z = ((1. / jnp.sqrt(2 * jnp.pi)) * jnp.exp(-.5*R**2))
  return X+mean, Y+mean, Z

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
  
@interact(a=(0,10),b=(0, 10),c=(0, 1.))
def domain_x(a,b,c):
  plt_plot_bivariate_normal_pdf(*jnp_bivariate_normal_pdf(a,b,c))
  
