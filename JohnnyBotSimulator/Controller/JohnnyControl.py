import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import sys
import os
import cvxpy as cp

# Go one level up to the parent directory of Project
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..')) 
sys.path.append(parent_dir)

from Estimation.PNT_Estimation import *
from Dynamics.JohnnyDynamics import *
from Plotting.Plotting import *

class JohnnyController:
    def __init__(self):
        self.kp = 1  # Proportional gain
        self.kd = 0.8  # Derivative gain
        self.control = jnp.array([0, 0])  # Control input [x_dot_dot, y_dot_dot]
        #self.estimator = PNT_Estimation()  # Estimator object
        self.dynamics = JohnnyDynamics()
        self.plotter = Plotting()
        self.lightsource = []
        self.dt = 0.01  # Time step
        self.t0 = 0  # Initial time
        self.tf = 20  # Final time
        self.state0 = jnp.array([0, 0, 0, 0])  # Initial state [x, y, x_dot, y_dot]
        self.state = self.state0  # Current state
        self.desired_state = jnp.array([10, 10, 0, 0])  # Desired state [x, y, x_dot, y_dot]
        self.states = []  # State trajectory
        self.desired_states = []  # Desired state trajectory
        self.controls = []  # Control trajectory
        self.time = []  # Time trajectory
        self.MPC_horizon = 30  # MPC horizon

        self.max_velocity = 2
        self.min_velocity = -2

        self.max_acceleration = 2
        self.min_acceleration = -2

        # Define Q and R matrices as class variables
        self.Q = np.diag([2, 2, 1, 1])  # Penalize position error more than velocity error
        self.R = np.diag([0.1, 0.1])  # Penalize control effort

    def control_Simple(self):
        # Calculate the control input using PD control
        position = self.state[0:2]
        velocity = self.state[2:4]
        desired_position = self.desired_state[0:2]
        desired_velocity = self.desired_state[2:4]
        position_error = desired_position - position
        velocity_error = desired_velocity - velocity
        control_input = self.kp * position_error + self.kd * velocity_error


        return control_input
    
    def simulate(self):
        # Simple IVP simulation with control_simple
        print("Simulating")
        self.states = [self.state0]
        self.desired_states = [self.desired_state]
        self.controls = [self.control]
        self.time = [self.t0]
        state = self.state0
        t = self.t0
        while t < self.tf:
            # Calculate control input
            control = self.control_Simple()
            self.controls.append(control)
            self.control = control

            # Integrate the dynamics
            state = solve_ivp(self.dynamics.double_integrator, (t, t + self.dt), state, args=(control,)).y[:, -1]
            self.states.append(state)
            self.state = state

            # Append the desired state for this time step
            self.desired_states.append(self.desired_state)


            # Update time
            t += self.dt
            self.time.append(t)


    def CVX_controller_double_integrator(self):
        """
        Solve a convex optimization problem to compute the optimal state and control trajectories
        for a double integrator system using Q and R matrices.
        """
        

        # Define system matrices
     
        
        # Time horizon
        time_horizon = int(self.tf / self.dt)
        self.time = np.linspace(self.t0, self.tf, time_horizon)

        self.desired_states = np.tile(self.desired_state, (time_horizon, 1))
        #self.lightsource = np.tile(jnp.array([self.plotter.source[0], self.plotter.source[0], 0, 0]), (time_horizon, 1))

        # Define optimization variables
        us = cp.Variable((2, time_horizon))  # Control inputs [x_dot_dot, y_dot_dot]
        states = cp.Variable((4, time_horizon))  # States [x, y, x_dot, y_dot]


        # Define constraints
        constraints = []

        # Initial state constraint
        constraints.append(states[:, 0] == self.state0)

        
        # Dynamics constraints
        for t in range(time_horizon - 1):
            constraints.append(states[:, t + 1] == states[:, t] + (self.dynamics.A @ states[:, t] + self.dynamics.B @ us[:, t])*self.dt)

        # Control input constraints
        constraints.append(us[0, :] <= self.max_acceleration)
        constraints.append(us[0, :] >= self.min_acceleration)
        constraints.append(us[1, :] <= self.max_acceleration)
        constraints.append(us[1, :] >= self.min_acceleration)

       # Define the objective function (vectorized)
        objective = cp.Minimize(
            cp.sum([cp.quad_form(states[:, t] - self.desired_states[t, :], self.Q) for t in range(self.MPC_horizon)]) +
            #.5*cp.sum([cp.quad_form(states[:, t] - self.lightsource[t, :], self.Q) for t in range(self.MPC_horizon)]) +
            cp.sum([cp.quad_form(us[:, t], self.R) for t in range(self.MPC_horizon - 1)])
        )

        # Solve the optimization problem
        prob = cp.Problem(objective, constraints)
        result = prob.solve()

        # Check if the problem was solved successfully
        if prob.status == cp.OPTIMAL:
            print("Optimization successful!")
            print("Optimal cost:", result)
            print("Optimal control inputs:", us.value.T)
            print("Optimal states:", states.value.T)
            self.states = states.value.T
            self.controls = us.value.T  
        else:
            print("Optimization failed:", prob.status)


    def CVX_controller_MPC(self):
        """
        Solve a convex optimization problem to compute the optimal state and control trajectories
        for a double integrator system using Q and R matrices with boundary constraints.
        """
        self.states = [self.state0]

        # Define optimization variables
        us = cp.Variable((2, self.MPC_horizon - 1))  # Control inputs [x_dot_dot, y_dot_dot]
        states = cp.Variable((4, self.MPC_horizon))  # States [x, y, x_dot, y_dot]

        state1 = cp.Parameter(4)

       
        time_horizon = int(self.tf / self.dt)
        self.time = np.linspace(self.t0, self.tf, time_horizon)
        self.desired_states = np.tile(self.desired_state, (time_horizon, 1))
        print("Desired states:", self.desired_states)

        self.lightsource = np.tile(jnp.array([self.plotter.source[0], self.plotter.source[0], 0, 0]), (time_horizon, 1))

        # Define constraints list
        constraints = []

        # Initial state constraint
        constraints.append(states[:, 0] == state1)

        # Dynamics constraints
        for t in range(self.MPC_horizon - 1):
            constraints.append(states[:, t + 1] == states[:, t] + (self.dynamics.A @ states[:, t] + self.dynamics.B @ us[:, t]) * self.dt)

        # Control input constraints
        constraints.append(us[0, :] <= self.max_acceleration)
        constraints.append(us[0, :] >= self.min_acceleration)
        constraints.append(us[1, :] <= self.max_acceleration)
        constraints.append(us[1, :] >= self.min_acceleration)

        # Define the objective function (vectorized)
        objective = cp.Minimize(
            cp.sum([cp.quad_form(states[:, t1] - self.desired_states[t1, :], self.Q)   for t1 in range(self.MPC_horizon)]) +
           # 100*cp.sum([cp.quad_form(states[:, t3] - self.lightsource[t3, :], self.Q) for t3 in range(self.MPC_horizon)]) +
            cp.sum([cp.quad_form(us[:, t2], self.R) for t2 in range(self.MPC_horizon - 1)])
        )

        # Solve the optimization problem
        prob = cp.Problem(objective, constraints)

        # Solve the problem iteratively
        for i in range(int(self.tf / self.dt) - 1):
            new_state = self.states[-1]
            print("New state:", new_state)
            state1.project_and_assign(new_state)

            print(state1.value)

            # Solve the optimization problem
            result = prob.solve()

            # Check if the problem was solved successfully
            if prob.status == cp.OPTIMAL:
                print(f"Step {i}: Optimization successful!")
                print("Optimal cost:", result)
                
                # Append the first state and control input to the trajectories
                self.states.append(states.value[:, 1])  # First state in the horizon
                self.controls.append(us.value[:, 0])  # First control input in the horizon

            else:
                print(f"Step {i}: Optimization failed:", prob.status)
                break

        
            
    def plot(self): 
        print("plotting")
        self.plotter.states = np.array(self.states)
        self.plotter.desired_states = np.array(self.desired_states)
        self.plotter.controls = np.array(self.controls)
        self.plotter.time = np.array(self.time)
        self.plotter.plot_states_controls()
        self.plotter.plot_trajectory()
        plt.show()


