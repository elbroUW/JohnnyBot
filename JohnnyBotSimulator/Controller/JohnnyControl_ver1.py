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

#from Estimation.PNT_Estimation import *
from Dynamics.JohnnyDynamics import *
from Plotting.Plotting import *

class JohnnyController:
    def __init__(self):
        self.kp = 1 # Proportional gain
        self.kd = .8 # Derivative gain
        self.control = jnp.array([0, 0])  # Control input [x_dot_dot, y_dot_dot]
        #self.estimator = PNT_Estimation()  # Estimator object
        self.dynamics = JohnnyDynamics()
        self.plotter = Plotting()
        self.dt = 0.01  # Time step
        self.t0 = 0  # Initial time
        self.tf = 10  # Final time
        self.state0 = jnp.array([0, 0, 1, 0])  # Initial state [x, y, x_dot, y_dot]
        self.state = self.state0
        self.desired_state = jnp.array([10, 10, 0, 0])  # Desired state [x, y, x_dot, y_dot]
        self.states = []  # State trajectory
        self.desired_states = []  # Desired state trajectory
        self.controls = []  # Control trajectory
        self.time = []  # Time trajectory

        self.max_velocity = 2
        self.min_velocity = -2

        self.max_acceleration = 2
        self.min_acceleration = -2

    def control_Simple(self):
        # Calculate the control input using PD control
        position = self.state[0:2]
        print("Position: ", position)
        velocity = self.state[2:4]
        desired_position = self.desired_state[0:2]
        desired_velocity = self.desired_state[2:4]
        position_error = desired_position - position
        print("Position error: ", position_error)
        velocity_error = desired_velocity - velocity
        control_input = self.kp * position_error + self.kd * velocity_error
        print("Control input: ", control_input)

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
        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
        
        B = np.array([[0, 0],
                      [0, 0],
                      [1, 0],
                      [0, 1]])
        
        # Time horizon
        time_horizon = int(self.tf / self.dt)
        self.time = np.linspace(self.t0, self.tf, time_horizon)

        self.desired_states = np.tile(self.desired_state, (time_horizon, 1))

        # Define optimization variables
        us = cp.Variable((2, time_horizon))  # Control inputs [x_dot_dot, y_dot_dot]
        states = cp.Variable((4, time_horizon))  # States [x, y, x_dot, y_dot]

        # Define Q and R matrices
        Q = np.diag([1, 1, 1, 1])  # Penalize position error more than velocity error
        R = np.diag([0.1, 0.1])  # Penalize control effort

        # Define constraints
        constraints = []

        # Initial state constraint
        constraints.append(states[:, 0] == self.state0)

        
        # Dynamics constraints
        for t in range(time_horizon - 1):
            constraints.append(states[:, t + 1] == states[:, t] + (A @ states[:, t] + B @ us[:, t])*self.dt)

        # Control input constraints
        constraints.append(us[0, :] <= self.max_acceleration)
        constraints.append(us[0, :] >= self.min_acceleration)
        constraints.append(us[1, :] <= self.max_acceleration)
        constraints.append(us[1, :] >= self.min_acceleration)

        # Define the objective function
        objective = cp.Minimize(
            cp.sum([cp.quad_form(states[:, t] - self.desired_state, Q) + cp.quad_form(us[:, t], R) for t in range(time_horizon)])
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


    def CVX_controller_double_integrator_boundary_constraint(self):
        """
        Solve a convex optimization problem to compute the optimal state and control trajectories
        for a double integrator system using Q and R matrices with boundary constraints.
        """

        # Define system matrices
        A = np.array([[0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
        
        B = np.array([[0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]])
        
        # Time horizon
        time_horizon = int(self.tf / self.dt)
        self.time = np.linspace(self.t0, self.tf, time_horizon)

        self.desired_states = np.tile(self.desired_state, (time_horizon, 1))

        # Define optimization variables
        us = cp.Variable((2, time_horizon))  # Control inputs [x_dot_dot, y_dot_dot]
        states = cp.Variable((4, time_horizon))  # States [x, y, x_dot, y_dot]

        # Define Q and R matrices
        Q = np.diag([5, 5, 1, 1])  # Penalize position error more than velocity error
        R = np.diag([0.1, 0.1])  # Penalize control effort

        # Define constraints list
        constraints = []

        # Initial state constraint
        constraints.append(states[:, 0] == self.state0)

        # Dynamics constraints
        for t in range(time_horizon - 1):
            constraints.append(states[:, t + 1] == states[:, t] + (A @ states[:, t] + B @ us[:, t]) * self.dt)

        
        # Control input constraints
        constraints.append(us[0, :] <= self.max_acceleration)
        constraints.append(us[0, :] >= self.min_acceleration)
        constraints.append(us[1, :] <= self.max_acceleration)
        constraints.append(us[1, :] >= self.min_acceleration)

        # Define the objective function (vectorized)
        objective = cp.Minimize(
            cp.sum([cp.quad_form(states[:, t] - self.desired_states[t, :], Q) for t in range(time_horizon)]) +
            cp.sum([cp.quad_form(us[:, t], R) for t in range(time_horizon)])
        )

        # Solve the optimization problem
        prob = cp.Problem(objective, constraints)
        result = prob.solve()

        # Check if the problem was solved successfully
        if prob.status == cp.OPTIMAL:
            print("Optimization successful!")
            print("Optimal cost:", result)
            self.states = states.value.T
            self.controls = us.value.T  
        else:
            print("Optimization failed:", prob.status)


    def CVX_controller_MPC(self):
        """

        """
        self.states.append(self.state0)
        self.controls.append(self.control)

        # Define system matrices
        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
        
        B = np.array([[0, 0],
                      [0, 0],
                      [1, 0],
                      [0, 1]])
        
        # Time horizon
        MPC_horizon = 30
        time_horizon = int(self.tf / self.dt)

        self.time = np.linspace(self.t0, self.tf, time_horizon)

        self.desired_states = np.tile(self.desired_state, (time_horizon, 1))

        # Define optimization variables
        us = cp.Variable((2, MPC_horizon - 1))  # Control inputs [x_dot_dot, y_dot_dot]
        states = cp.Variable((4, MPC_horizon))  # States [x, y, x_dot, y_dot]

        state1 = cp.Parameter(4)


        As = cp.Parameter((4, 4)) # for later
        As.project_and_assign(A)
        Bs = cp.Parameter((4, 2)) # for later  
        Bs.project_and_assign(B)


        # Define Q and R matrices
        Q = np.diag([2, 2, 1, 1])  # Penalize position error more than velocity error
        R = np.diag([0.1, 0.1])  # Penalize control effort

        # Define constraints list
        constraints = []

        # Initial state constraint
        constraints.append(states[:, 0] == state1)

        # Dynamics constraints
        for t in range(MPC_horizon - 1):
            constraints.append(states[:, t + 1] == states[:, t] + (A @ states[:, t] + B @ us[:, t]) * self.dt)

        # Control input constraints
        constraints.append(us[0, :] <= self.max_acceleration)
        constraints.append(us[0, :] >= self.min_acceleration)
        constraints.append(us[1, :] <= self.max_acceleration)
        constraints.append(us[1, :] >= self.min_acceleration)

        # Define the objective function (vectorized)
        objective = cp.Minimize(
            cp.sum([cp.quad_form(states[:, t] - self.desired_states[t, :], Q) for t in range(MPC_horizon)]) +
            cp.sum([cp.quad_form(us[:, t], R) for t in range(MPC_horizon - 1)])
        )

        # Solve the optimization problem
        prob = cp.Problem(objective, constraints)

        # Solve the problem iteratively
        for i in range(time_horizon-1):

            new_state = self.states[-1]
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



    def CVX_controller_MPC_Experiments(self):
        """

        """
        self.states.append(self.state0)
        self.controls.append(self.control)

        # Define system matrices
        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
        
        B = np.array([[0, 0],
                      [0, 0],
                      [1, 0],
                      [0, 1]])
        
        # Time horizon
        MPC_horizon = 30
        time_horizon = int(self.tf / self.dt)

        self.time = np.linspace(self.t0, self.tf, time_horizon)

        self.desired_states = np.tile(self.desired_state, (time_horizon, 1))

        # Define optimization variables
        us = cp.Variable((2, MPC_horizon - 1))  # Control inputs [x_dot_dot, y_dot_dot]
        states = cp.Variable((4, MPC_horizon))  # States [x, y, x_dot, y_dot]

        state1 = cp.Parameter(4)



        As = cp.Parameter((4, 4)) # for later
        As.project_and_assign(A)
        Bs = cp.Parameter((4, 2)) # for later  
        Bs.project_and_assign(B)


        # Define Q and R matrices
        Q = np.diag([2, 2, 1, 1])  # Penalize position error more than velocity error
        R = np.diag([0.1, 0.1])  # Penalize control effort

        # Define constraints list
        constraints = []

        # Initial state constraint
        constraints.append(states[:, 0] == state1)

        # Dynamics constraints
        for t in range(MPC_horizon - 1):
            constraints.append(states[:, t + 1] == states[:, t] + (A @ states[:, t] + B @ us[:, t]) * self.dt)
            
        

        # Control input constraints
        constraints.append(us[0, :] <= self.max_acceleration)
        constraints.append(us[0, :] >= self.min_acceleration)
        constraints.append(us[1, :] <= self.max_acceleration)
        constraints.append(us[1, :] >= self.min_acceleration)
       
       

        # Define the objective function (vectorized)
        objective = cp.Minimize(
            cp.sum([cp.quad_form(states[:, t] - self.desired_states[t, :], Q) for t in range(MPC_horizon)]) +
            cp.sum([cp.quad_form(us[:, t], R) for t in range(MPC_horizon - 1)])
        )

        # Solve the optimization problem
        prob = cp.Problem(objective, constraints)

        # Solve the problem iteratively
        for i in range(time_horizon-1):

            new_state = self.states[-1]
            state1.project_and_assign(new_state)
             # b CBF constraint
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


        