import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Go one level up to the parent directory of Project
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

class Plotting:
    def __init__(self) -> None:
        self.states = [] # state is [x, y, x_dot, y_dot]
        self.desired_states = [] # desired state is [x, y, x_dot, y_dot]
        self.controls = []  # control is [x_dot_dot, y_dot_dot]
        self.time = []
        self.dt = 0.01
        self.t0 = 0
        self.tf = 10
        self.source = [0, 0]
        self.source_gain = 10
        self.obstacles_bool = False
        self.obstacles = [] #obstacle is consisted of a list of [x, y, radius]
        

    # plot trajectory on x and y axis with simulated light-source that spreads out from a point 
    # with a decreasing intensity shown as shading of the color
    def plot_trajectory(self):
        fig, ax = plt.subplots()

        # Create a grid of points
        x = np.linspace(min(self.states[:, 0]) - 1, max(self.states[:, 0]) + 1, 100)
        y = np.linspace(min(self.states[:, 1]) - 1, max(self.states[:, 1]) + 1, 100)
        X, Y = np.meshgrid(x, y)

        if self.obstacles_bool:
            for obs in self.obstacles:
                circle = plt.Circle((obs[0], obs[1]), obs[2], color='b', fill=False)
                ax.add_artist(circle)

        # Calculate the distance from the source
        dist = np.sqrt((X - self.source[0])**2 + (Y - self.source[1])**2)

        # Calculate the intensity based on the distance
        intensity = np.exp(-dist**2 / (2 * self.source_gain**2))

        # Plot the intensity as a background
        ax.imshow(intensity, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='hot', alpha=0.5)

        # Plot the trajectory
        ax.scatter(self.states[:, 0], self.states[:, 1], c=self.time,  label='Trajectory')
        ax.plot(self.source[0], self.source[1], 'ro', label='Light Source')
        ax.plot(self.desired_states[-1, 0], self.desired_states[-1, 1], 'g+', label='goal')
        ax.plot(self.states[0, 0], self.states[0, 1], 'bx', markersize = 10, label='start')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Trajectory of the robot')
        ax.legend()
        plt.show()


    # plot the states and controls
    def plot_states_controls(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Plot the states
        axs[0, 0].plot(self.time, self.states[:, 0], label='x')
        axs[0, 0].plot(self.time, self.desired_states[:, 0], label='x_desired')
        axs[0, 0].set_xlabel('Time (s)')
        axs[0, 0].set_ylabel('X')
        axs[0, 0].set_title('X vs Time')
        axs[0, 0].legend()

        axs[0, 1].plot(self.time, self.states[:, 1], label='y')
        axs[0, 1].plot(self.time, self.desired_states[:, 1], label='y_desired')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Y')
        axs[0, 1].set_title('Y vs Time')
        axs[0, 1].legend()

        axs[1, 0].plot(self.time, self.states[:, 2], label='x_dot')
        axs[1, 0].plot(self.time, self.desired_states[:, 2], label='x_dot_desired')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('X_dot')
        axs[1, 0].set_title('X_dot vs Time')
        axs[1, 0].legend()

        axs[1, 1].plot(self.time, self.states[:, 3], label='y_dot')
        axs[1, 1].plot(self.time, self.desired_states[:, 3], label='y_dot_desired')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Y_dot')
        axs[1, 1].set_title('Y_dot vs Time')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Plot the controls
        axs[0].plot(self.time, self.controls[:, 0], label='x_dot_dot')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('X_dot_dot')
        axs[0].set_title('X_dot_dot vs Time')
        axs[0].legend()

        axs[1].plot(self.time, self.controls[:, 1], label='y_dot_dot')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Y_dot_dot')
        axs[1].set_title('Y_dot_dot vs Time')
        axs[1].legend()