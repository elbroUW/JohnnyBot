import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy.integrate import solve_ivp
from scipy import integrate
import jax
import jax.numpy as jnp
import cvxpy as cp
from jax.typing import ArrayLike
from jax import jit
import random as rand
import plotly.graph_objects as go
import time 



class Estimation:
    def __init__(self) -> None:
        self.light_source = [0, 0]
        self.light_source_gain = 10     
        pass


    def gen_light_noise_linear(self, state):
        """
        Generate light source noise
        Inputs:
            state: A jax.numpy array of size (n,) representing [x, y, x_dot, y_dot]
        Output:
            light_noise: A jax.numpy array of size (n,) representing [x_noise, y_noise, x_dot_noise, y_dot_noise]
        """
        x, y, x_dot, y_dot = state
        Noise_x = x- self.light_source[0]
        Noise_y = y- self.light_source[1]
        light_noise = jnp.array([rand.gauss(0, Noise_x/10) + x, rand.gauss(0, Noise_y/10)+ y, x_dot, y_dot])
        return light_noise
    

    
