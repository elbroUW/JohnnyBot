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




class JohnnyBot:
    def __init__(self) -> None:
        # Initialize control input (angular_velocity, linear_velocity)
        self.control = jnp.array([0, 0])

        # Define the goal state [x_goal, y_goal, theta_goal, v_goal]
        self.goalstate = jnp.array([0, 0, 0, 0])

        # Define the initial state [x_init, y_init, theta_init, v_init]
        self.initstate = jnp.array([2, 2, 0, 0])
        self.state = self.initstate
        
        #set initial estimate to init state
        self.est_state = self.initstate

        self.est_his = [self.est_state]
        self.state_his = [self.state]
        self.theta_his = [self.state[2]]

        self.distance_noise_std = 0.1

        self.angle_noise_std = 0.05

        self.sensor_pos =  jnp.array([[-20, 20, 15], 
                                     [20, 17.5, 10], 
                                     [18.4, -5.5, 20], 
                                     # [10,10,-15], 
                                                ]) # Sensor positions

        # Simulation time step and time span
        self.dt = 0.1
        self.t_span = (0, 20)
        self.currtime = 0

        # Controller gain parameters for linear and angular velocities
        self.kv = 0.1   # Proportional gain for linear velocity
        self.kw = 1.5   # Proportional gain for angular velocity

        # Maximum forward speed limit
        self.maxspeed = 10

        self.sol =[]


        # Parameters for trajectory arrow plotting
        self.arrowsize = 0.02
        self.arrow_step = 100

        pass

    def basic_controller(self, state):
        """
        Compute control inputs (angular_velocity, linear_velocity) based on the 
        current state and the goal state using a proportional controller.
        """
        # Unpack current state: [x, y, theta, v]
        x, y, theta, v = state

        # Unpack goal state: [x_goal, y_goal, theta_goal, _]
        x_goal, y_goal, theta_goal, _ = self.goalstate

        # Compute position error
        dx = x_goal - x
        dy = y_goal - y
        distance_to_goal = jnp.sqrt(dx**2 + dy**2)

        # Compute the desired heading angle to reach the goal
        desired_heading = jnp.arctan2(dy, dx)

        # Compute orientation error (difference between desired and current heading)
        heading_error = desired_heading - theta

        # Normalize heading_error to the range [-pi, pi]
        heading_error = (heading_error + jnp.pi) % (2 * jnp.pi) - jnp.pi

        # Compute the linear velocity command:
        # Proportional to the distance to the goal and capped by the maximum speed
        v_cmd = self.kv * distance_to_goal
        v_cmd = jnp.clip(v_cmd, -self.maxspeed, self.maxspeed)

        # Compute the angular velocity command:
        # Proportional to the orientation (heading) error
        w_cmd = self.kw * heading_error

        # Update the control input (angular_velocity, linear_velocity)
        self.control = jnp.array([w_cmd, v_cmd])

    def state_derivative(self, t, state):
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
        v_dot = self.control[1]        # Linear velocity command
        theta = state[2]

        # The "v" in the state derivative is simply the commanded v_dot here
        v = v_dot

        # Compute derivatives of the state based on the current orientation and velocity
        x_dot = v * jnp.cos(theta)
        y_dot = v * jnp.sin(theta)

        # Construct and return the state derivative vector
        state_derivative = jnp.array([x_dot, y_dot, theta_dot, v_dot])
        return state_derivative
    
    def state_derivative_Noisy(self, t, state):
        """
        Computes the state derivative x_dot = f(x, u), where x is the state and u is the control.
        Inputs:
            t      : Current time (not used directly in this controller)
            state  : A jax.numpy array of size (n,) representing [x, y, theta, v]
        Output:
            state_derivative : A jax.numpy array of size (n,) representing [x_dot, y_dot, theta_dot, v_dot]
        """
        #compute Noisy Position estimate
        
        self.est_state = give_noisy_pos_est(state, self.est_state, self.sensor_pos, self.distance_noise_std)
        

        #compute Noisy Angle estimate
        self.est_state = give_noisy_angle_est(self.est_state, self.angle_noise_std)

        self.est_his.append(self.est_state)
        
       
        print("estimate =", self.est_state)
        print("true state = ", state)
        print("time =", t)
       
        # First, compute the control inputs based on the current state
        self.basic_controller(self.est_state)

        # Extract necessary values
        theta_dot = self.control[0]    # Angular velocity command
        v_dot = self.control[1]        # Linear velocity command
        theta = state[2]

        print("theta = ", theta)

        
        # The "v" in the state derivative is simply the commanded v_dot here
        v = v_dot

        # Compute derivatives of the state based on the current orientation and velocity
        x_dot = v * jnp.cos(theta)
        y_dot = v * jnp.sin(theta)


        # Construct and return the state derivative vector
        state_derivative = jnp.array([x_dot, y_dot, theta_dot, v_dot])

        return state_derivative
    


    def plotTraj(self):
        """
        Plot the trajectory of the robot over time and indicate orientation with arrows.
        Inputs:
            sol : The solution object returned by solve_ivp containing time and state arrays
        """
        # Extract time and state from the solution
        #t = self.sol.t
        #state = self.sol.y
        # Extract time and state from the solution
        t = np.linspace(0,self.t_span[1], int(self.t_span[1]/self.dt))

        
        # Create a figure for trajectory plotting
        plt.figure(figsize=(5, 5))

        # Add orientation arrows along the path at regular intervals
        
        for i in range(0, len(t), self.arrow_step):
            dx = np.cos(jnp.stack(self.state_his).T[2, i]) * 0.5  # Arrow length along x-direction
            dy = np.sin(jnp.stack(self.state_his).T[2, i]) * 0.5  # Arrow length along y-direction
            plt.arrow(jnp.stack(self.state_his).T[0, i], jnp.stack(self.state_his).T[1, i], dx, dy,
                      head_width=self.arrowsize, head_length=self.arrowsize, color="red")
        

        # Mark the start position
        plt.scatter(self.initstate[0], self.initstate[1], color="green", label="Start")

        # Mark the goal position
        x_goal, y_goal, _, _ = self.goalstate
        plt.scatter(x_goal, y_goal, color="red", label="Goal")

        # Plot the full trajectory
        #plt.plot(state[0, :], state[1, :], 'b-', label='Trajectory')
        plt.plot(jnp.stack(self.state_his).T[0], jnp.stack(self.state_his).T[1],'b-', label='traj')
        plt.scatter(jnp.stack(self.est_his).T[0], jnp.stack(self.est_his).T[1], label='est traj')

        # Add plot labels, title, and legend
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Trajectory')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    def plotXY(self):
        """
        Plot the X and Y positions over time as separate subplots.
        Inputs:
            sol : The solution object containing time and state arrays
        """
        # Extract time and state from the solution
        t = np.linspace(0,self.t_span[1], int(self.t_span[1]/self.dt))

        # Create a figure with two subplots: one for X over time, one for Y over time
        plt.figure(figsize=(15, 4))

        # Plot X position over time
        plt.subplot(1, 3, 1)
        plt.plot(t, jnp.stack(self.state_his).T[0], 'r', label='X Position')
        plt.plot(t, jnp.stack(self.est_his).T[0], 'b-', label='X Est Position')
        plt.xlabel('Time (sec)')
        plt.ylabel('X Position')
        plt.title('X Position over Time')
        plt.legend()
        plt.grid(True)
       

        # Plot Y position over time
        plt.subplot(1, 3, 2)
        plt.plot(t, jnp.stack(self.state_his).T[1], 'r', label='Y Position')
        plt.plot(t, jnp.stack(self.est_his).T[1], 'b-', label='Y Est Position')
        plt.xlabel('Time (sec)')
        plt.ylabel('Y Position')
        plt.title('Y Position over Time')
        plt.legend()
        plt.grid(True)
       

        # Plot theta  over time
        plt.subplot(1, 3, 3)
        plt.plot(t, jnp.stack(self.state_his).T[2] * 180/np.pi, 'r', label='Theta')
        plt.plot(t, jnp.stack(self.est_his).T[2] * 180/np.pi, 'b-', label='est Theta')
        plt.xlabel('Time (sec)')
        plt.ylabel('Theta')
        plt.title('theta over Time')
        plt.legend()
        plt.grid(True)
        

        # Show the plots
        plt.show()

    # currently doesn't work
    def runsim(self):
        """
        Run the simulation by integrating the state derivatives over the defined time span.
        Then plot the resulting trajectory and X/Y position profiles.
        """
        # Solve the ODE using the defined state_derivative function  min_step = .01, max_step = self.dt
        sol = solve_ivp(self.state_derivative_Noisy, self.t_span, self.initstate, rtol=1e-1, atol=1e-6, method = 'LSODA')
        self.sol = sol
        # Plot the resulting trajectory and position over time
        self.plotTraj()
        #self.plotXY()

    def runsim2(self):
        """
        Run the simulation by integrating the state derivatives over the defined time span.
        Then plot the resulting trajectory and X/Y position profiles.
        """
        # Solve the ODE using the defined state_derivative function
        for i in range(int(self.t_span[1]/self.dt)-1):
            self.currtime= self.dt +self.currtime
            derivative = self.state_derivative_Noisy(self.currtime, self.state)
            self.state = derivative*self.dt + self.state
            self.state_his.append(self.state)
            #print("deriv = ",derivative)
            
        
        # Plot the resulting trajectory and position over time
        self.plotTraj()
        self.plotXY()



