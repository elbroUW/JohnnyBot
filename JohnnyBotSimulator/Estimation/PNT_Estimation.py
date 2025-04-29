import numpy as np
import jax.numpy as jnp
import sys
import os


# Go one level up to the parent directory of Project
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..')) 
sys.path.append(parent_dir)
from Dilution_of_Precision import *
from Plotting.Estimation_plotting import *

class PNT_Estimation:
    """
    A class for Position, Navigation, and Timing (PNT) estimation using noisy sensor data.
    """

    def __init__(self, sensor_positions, noise_std=1.0, max_iterations=100, tolerance=1e-6):
        """
        Initialize the PNT_Estimation class.

        Args:
            sensor_positions (np.ndarray): Array of sensor positions with shape (m, 3).
            noise_std (float): Standard deviation of Gaussian noise for distance measurements.
            max_iterations (int): Maximum number of iterations for gradient descent.
            tolerance (float): Convergence threshold for gradient descent.
        """
        self.sensor_positions = np.array(sensor_positions)
        self.noise_std = noise_std
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.plotter = EstimationPlotting()

    def generate_noisy_distances(self, true_position):
        """
        Generate noisy distance measurements from sensors to the true position.

        Args:
            true_position (np.ndarray): True position of the object (3D).

        Returns:
            np.ndarray: Noisy distance measurements from each sensor.
        """
        true_distances = np.linalg.norm(self.sensor_positions - true_position, axis=1)
        noise = np.random.normal(0, self.noise_std, size=true_distances.shape)
        noisy_distances = true_distances + noise
        return noisy_distances

    def gradient_descent_position_estimation(self, measured_distances, initial_guess):
        """
        Estimate the object's position using gradient descent.

        Args:
            measured_distances (np.ndarray): Noisy distance measurements from sensors.
            initial_guess (np.ndarray): Initial guess for the object's position (3D).

        Returns:
            np.ndarray: Estimated position of the object (3D).
        """
        x = initial_guess.copy()
        m = self.sensor_positions.shape[0]

        for iteration in range(self.max_iterations):
            A = np.zeros((m, 3))
            d = np.zeros(m)

            for i in range(m):
                s_i = self.sensor_positions[i]
                r_i = measured_distances[i]
                diff = x - s_i
                distance = np.linalg.norm(diff)

                if distance == 0:
                    continue  # Avoid division by zero

                # Residual and unit vector
                d[i] = r_i - distance
                A[i] = diff / distance

            # Solve the least squares problem
            ATA = A.T @ A
            try:
                A_plus = np.linalg.inv(ATA) @ A.T
            except np.linalg.LinAlgError:
                print("Singular matrix encountered during gradient descent.")
                return x

            delta_x = A_plus @ d
            x_new = x + delta_x

            # Check for convergence
            if np.linalg.norm(x_new - x) < self.tolerance:
                #print(f"Converged in {iteration + 1} iterations.")
                return x_new

            x = x_new

        print("Maximum iterations reached without convergence.")
        return x

    def estimate_position(self, true_position, initial_guess):
        """
        Estimate the object's position given a true position and an initial guess.

        Args:
            true_position (np.ndarray): True position of the object (3D).
            initial_guess (np.ndarray): Initial guess for the object's position (3D).

        Returns:
            dict: A dictionary containing:
                - "true_position": The true position of the object.
                - "noisy_distances": The noisy distance measurements.
                - "estimated_position": The estimated position of the object.
        """
        true_position = np.array(true_position)
        initial_guess = np.array(initial_guess)

        # Generate noisy distances
        noisy_distances = self.generate_noisy_distances(true_position)

        # Estimate position using gradient descent
        estimated_position = self.gradient_descent_position_estimation(noisy_distances, initial_guess)

        return estimated_position[0:2]
    
    def estimate_positions(self, true_position, initial_guess, num_samples):
        """
        Estimate multiple positions given true positions and an initial guess.

        Args:
            true_positions (list of np.ndarray): List of true positions of the object (3D).
            initial_guess (np.ndarray): Initial guess for the object's position (3D).
            num_samples (int): Number of samples to generate.

        Returns:
            list: A list of dictionaries containing:
                - "true_position": The true position of the object.
                - "noisy_distances": The noisy distance measurements.
                - "estimated_position": The estimated position of the object.
        """
        results = []
        for i in range(num_samples):
            result = self.estimate_position(true_position, initial_guess)
            results.append(result)
        return results





