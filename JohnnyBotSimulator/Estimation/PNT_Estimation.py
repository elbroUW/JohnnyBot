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




def generate_noisy_distances(sensor_positions, true_position, noise_std):
    """
    Generate simulated noisy distance measurements from sensors to an object.

    Parameters:
    - sensor_positions: numpy array of shape (m, 3), positions of m sensors.
    - true_position: numpy array of shape (3,), true position of the object.
    - noise_std: float or numpy array of shape (m,), standard deviation of Gaussian noise.

    Returns:
    - noisy_distances: numpy array of shape (m,), simulated noisy distance measurements.
    """
    #print(true_position)
    #print(sensor_positions)
    #print(sensor_positions - true_position)

    # Calculate true distances from each sensor to the object
    true_distances = jnp.linalg.norm(sensor_positions - true_position, axis=1)

    # Generate Gaussian noise
    if isinstance(noise_std, (float, int)):
        noise = np.random.normal(0, noise_std, size=true_distances.shape)
    else:
        noise = np.random.normal(0, noise_std)

    # Add noise to the true distances
    noisy_distances = true_distances + noise
    #print(noisy_distances)
    return noisy_distances

def compute_distances(sensor_positions, object_position):
    """
    Compute the Euclidean distances from an object to each sensor.

    Parameters:
    - sensor_positions: numpy array of shape (m, n), positions of m sensors in n-dimensional space.
    - object_position: numpy array of shape (n,), position of the object in n-dimensional space.

    Returns:
    - distances: numpy array of shape (m,), distances from the object to each sensor.
    """
    # Calculate distances
    distances = np.linalg.norm(sensor_positions - object_position, axis=1)
    return distances




def gradient_descent_position_estimation(sensor_positions, measured_distances, initial_guess,
                                         max_iterations=100, tolerance=1e-6):
    """
    Estimate the object's position using gradient descent.
    
    Parameters:
    - sensor_positions: numpy array of shape (m, 3)
    - measured_distances: numpy array of shape (m,)
    - initial_guess: numpy array of shape (3,)
    - max_iterations: int, maximum number of iterations
    - tolerance: float, convergence threshold
    
    Returns:
    - estimated_position: numpy array of shape (3,), estimated object position
    """
    x = initial_guess.copy()
    m = sensor_positions.shape[0]
 
    
    estm_pos = []
    for iteration in range(max_iterations):
        # Initialize gradient
        a = []
        A = np.zeros((m, 3))
        d = np.zeros(m)
        for i in range(m):
            s_i = sensor_positions[i]
            r_i = measured_distances[i]
            
            # Vector from sensor to current estimate
            #print("x: ", x)
            #print("s_i", s_i)
            diff = x - s_i
            distance = np.linalg.norm(diff)
            #print(distance)
            
            # Avoid division by zero
            if distance == 0:
                print("bad zero")
                continue  # Or handle appropriately
             
            
            # Residual
            f_i = r_i - distance
            #print("\ndi:\n", f_i)
            
            d[i] = f_i
            
            # Unit vector
            u_i = diff / distance
            #print(u_i)
            
            A[i] = u_i

        
        ATA = A.T @ A
     
        A_plus_alt = np.linalg.inv(ATA) @ A.T

        # Compute estimated position
        estimated_position = A_plus_alt @ d

        # Update position estimate
        x_new = x + estimated_position   # Negative gradient since we are minimizing

       
        # Check for convergence
        if np.linalg.norm(x_new - x) < tolerance:
            print(f"Converged in {iteration + 1} iterations.")
            return x_new #cost_history, residuals, estm_pos, A
        
        # Update estimate
        x = x_new
        

    print(x)
    #print("Maximum iterations reached without convergence.")
    return x #cost_history, residuals, estm_pos, A


def give_noisy_pos_est(state, initial_guess, sensor_Pos, noise_std):
    x,y,_,_ = state
    x_1, y_1,_,_ = initial_guess
    Pos_true = np.array([x,y,0])
    guess = jnp.array([x_1, y_1,0])
    
    noisy_distances = generate_noisy_distances(sensor_Pos, Pos_true, noise_std)

    est_pos= gradient_descent_position_estimation(sensor_Pos, noisy_distances,guess ,
                                    max_iterations=1000, tolerance=1e-6)
    
    
    est_state = jnp.array([est_pos[0], est_pos[1], state[2], state[3]])

    return est_state


# add noise to the angle 
def give_noisy_angle_est(state, noise_std):
    x,y,theta,omega = state
    angle_noisy = theta + np.random.normal(0, noise_std)

    return jnp.array([x,y,angle_noisy,omega])





