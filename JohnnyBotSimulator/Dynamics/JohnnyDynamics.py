import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sys
import os

# Go one level up to the parent directory of Project
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from Estimation.PNT_Estimation import *

class JohnnyDynamics:
    def __init__(self) -> None:

        self.state = jnp.array([0, 0, 0, 0])  # State vector [x, y, x_dot, y_dot]
        self.control = jnp.array([0, 0])  # Control input [x_dot_dot, y_dot_dot]
        pass

    def double_integrator(self, t, state, control):
        """
        Computes the state derivative x_dot = f(x, u), where x is the state and u is the control.
        Inputs:
            t      : Current time (not used directly in this controller)
            state  : A jax.numpy array of size (n,) representing [x, y, x_dot, y_dot]
            control: A jax.numpy array of size (m,) representing [x_dot_dot, y_dot_dot]
        Output:
            state_derivative : A jax.numpy array of size (n,) representing [x_dot, y_dot, x_dot_dot, y_dot_dot]
        
        State-space representation:
        The state vector is defined as:
        state = [x, y, x_dot, y_dot]
        
        The control input vector is defined as:
        control = [x_dot_dot, y_dot_dot]
        
        The state-space equations are:
        [ x_dot     ]   [ 0  0  1  0 ] [ x       ]   [ 0  0 ] [ x_dot_dot     ]
        [ y_dot     ] = [ 0  0  0  1 ] [ y       ] + [ 0  0 ] [ y_dot_dot     ]
        [ x_dot_dot ]   [ 0  0  0  0 ] [ x_dot   ]   [ 1  0 ] [               ]
        [ y_dot_dot ]   [ 0  0  0  0 ] [ y_dot   ]   [ 0  1 ] [               ]
        
        This can be written in matrix form as:
        state_derivative = A @ state + B @ control
        """

        A = jnp.array([[0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])
        
        B = jnp.array([[0, 0],
                       [0, 0],
                       [1, 0],
                       [0, 1]])
        
        state_derivative = A @ state + B @ control
        #print("State derivative: ")
        #print(state_derivative)

        
        return state_derivative
    
    

    def dubin_Car_Dynamics(self, t, state):
        """
        Computes the state derivative x_dot = f(x, u), where x is the state and u is the control.
        Inputs:
            t      : Current time (not used directly in this controller)
            state  : A jax.numpy array of size (n,) representing [x, y, theta, v]
        Output:
            state_derivative : A jax.numpy array of size (n,) representing [x_dot, y_dot, theta_dot, v_dot]
        """
        # First, compute the control inputs based on the current state
        self.basic_controller(state)

        # Extract necessary values
        theta_dot = self.control[0]    # Angular velocity command
        v_dot = self.control[1]        # Linear accelra command
        theta = state[2]

        # The "v" in the state derivative is simply the commanded v_dot here
        v = v_dot

        # Compute derivatives of the state based on the current orientation and velocity
        x_dot = v * jnp.cos(theta)
        y_dot = v * jnp.sin(theta)

        # Construct and return the state derivative vector
        state_derivative = jnp.array([x_dot, y_dot, theta_dot, v_dot])
        return state_derivative
