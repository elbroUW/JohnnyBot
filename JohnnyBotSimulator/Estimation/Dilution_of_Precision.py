import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt


class DilutionOfPrecision:
    """
    Class to calculate and plot the Dilution of Precision (DOP) for a given set of sensors.
    """

    def __init__(self, sensors, bounds=None, grid_resolution=1, norm=True, invert_color=False, max_color=30):
        """
        Initialize the DOP class with sensor positions and configuration options.
        Args:
            sensors (np.ndarray): Array of sensor positions with shape (n_sensors, 3).
            bounds (tuple): (x_min, x_max, y_min, y_max, z_min, z_max). Default is (-20, 20, -20, 20, 0, 10).
            grid_resolution (float): Resolution of the grid. Default is 5.
            norm (bool): Whether to normalize HDOP values for elevation. Default is True.
            invert_color (bool): Whether to invert the color scale. Default is False.
        """
        self.sensors = np.array(sensors)
        self.bounds = bounds if bounds else (-20, 20, -30, 30, -5, 20)
        self.grid_resolution = grid_resolution
        self.norm = norm
        self.invert_color = invert_color
        self.color_style = 'YlGnBu' if not invert_color else 'RdYlBu'
        self.max_color = max_color

    def calculate_geometry_matrix(self, point):
        """
        Calculate the geometry matrix G for a given point based on sensor positions.
        Args:
            point (np.ndarray): A 3D point [x, y, z].
        Returns:
            np.ndarray: Geometry matrix G.
        """
        G = []
        for sensor in self.sensors:
            dx, dy, dz = sensor - point
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            if distance == 0:
                continue  # Avoid division by zero
            G.append([dx / distance, dy / distance, dz / distance])
        return np.array(G)

    def calculate_dop_values(self, G):
        """
        Calculate PDOP, HDOP, and VDOP from the geometry matrix G.
        Args:
            G (np.ndarray): Geometry matrix.
        Returns:
            tuple: PDOP, HDOP, and VDOP values.
        """
        try:
            Q = np.linalg.inv(G.T @ G)
            pdop = np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])
            hdop = np.sqrt(Q[0, 0] + Q[1, 1])
            vdop = np.sqrt(Q[2, 2])
            return pdop, hdop, vdop
        except np.linalg.LinAlgError:
            # If matrix is singular (poor geometry), return high DOP values
            return float('inf'), float('inf'), float('inf')

    def generate_dop_grid(self):
        """
        Generate DOP values over a 3D grid within the specified bounds.
        Returns:
            tuple: x_vals, y_vals, z_vals, pdop_grid, hdop_grid, vdop_grid.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        x_vals = np.arange(x_min, x_max + self.grid_resolution, self.grid_resolution)
        y_vals = np.arange(y_min, y_max + self.grid_resolution, self.grid_resolution)
        z_vals = np.arange(z_min, z_max + self.grid_resolution, self.grid_resolution)

        pdop_grid = np.zeros((len(x_vals), len(y_vals), len(z_vals)))
        hdop_grid = np.zeros((len(x_vals), len(y_vals), len(z_vals)))
        vdop_grid = np.zeros((len(x_vals), len(y_vals), len(z_vals)))

        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                for k, z in enumerate(z_vals):
                    point = np.array([x, y, z])
                    G = self.calculate_geometry_matrix(point)
                    pdop, hdop, vdop = self.calculate_dop_values(G)
                    pdop_grid[i, j, k] = pdop
                    hdop_grid[i, j, k] = hdop
                    vdop_grid[i, j, k] = vdop

        return x_vals, y_vals, z_vals, pdop_grid, hdop_grid, vdop_grid
    
    def plot_hdop_z_val(self):
        """
        Plot the HDOP values on a 2D plane (z = 0) using a 3D scatter plot.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        x_vals = np.arange(x_min, x_max + self.grid_resolution, self.grid_resolution)
        y_vals = np.arange(y_min, y_max + self.grid_resolution, self.grid_resolution)
        z_fixed = 0  # Fixed z position for the bottom of the box

        hdop_grid = np.zeros((len(x_vals), len(y_vals)))

        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                point = np.array([x, y, z_fixed])
                G = self.calculate_geometry_matrix(point)
                _, hdop, _ = self.calculate_dop_values(G)
                hdop_grid[i, j] = hdop

        x_flat = np.repeat(x_vals, len(y_vals))
        y_flat = np.tile(y_vals, len(x_vals))
        z_flat = hdop_grid.ravel()  # Use HDOP values for z position
        hdop_flat = hdop_grid.ravel()

        color_Hdop = np.clip(hdop_flat, 0, self.max_color)
        color_val = 1 / (color_Hdop**2 + 1e-6) if self.invert_color else color_Hdop
        z_max = np.max(hdop_flat) if np.max(hdop_flat) < 100 else 100

        fig = go.Figure(data=go.Scatter3d(
            x=x_flat,
            y=y_flat,
            z=z_flat,  
            mode='markers',
            marker=dict(
                size=2,
                color=color_val,
                colorscale=self.color_style,
                colorbar=dict(title="HDOP"),
                opacity=0.8
            )
        ))


        fig.update_layout(
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position',
                zaxis=dict(range=[0, z_max])
            ),
            title="3D HDOP Distribution on Bottom Plane of Box"
        )

        fig.show()
        
    def plot_hdop_2d(self):
        """
        Plot the HDOP values on a 2D plane (z = 0) using a 3D scatter plot.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        x_vals = np.arange(x_min, x_max + self.grid_resolution, self.grid_resolution)
        y_vals = np.arange(y_min, y_max + self.grid_resolution, self.grid_resolution)
        z_fixed = 0  # Fixed z position for the bottom of the box

        hdop_grid = np.zeros((len(x_vals), len(y_vals)))

        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                point = np.array([x, y, z_fixed])
                G = self.calculate_geometry_matrix(point)
                _, hdop, _ = self.calculate_dop_values(G)
                hdop_grid[i, j] = hdop

        x_flat = np.repeat(x_vals, len(y_vals))
        y_flat = np.tile(y_vals, len(x_vals))
        z_flat = np.full_like(x_flat, z_fixed)
        hdop_flat = hdop_grid.ravel()

        color_Hdop = np.clip(hdop_flat, 0, self.max_color)
        color_val = 1 / (color_Hdop**2 + 1e-6) if self.invert_color else color_Hdop

        


        fig = go.Figure(data=go.Scatter3d(
            x=x_flat,
            y=y_flat,
            z=z_flat,  # Fixed z position
            mode='markers',
            marker=dict(
                size=2,
                color=color_val,
                colorscale=self.color_style,
                colorbar=dict(title="HDOP"),
                opacity=0.8
            )
        ))

        fig.add_trace(go.Scatter3d(
            x=self.sensors[:, 0],
            y=self.sensors[:, 1],
            z=self.sensors[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond',
                opacity=1
            ),
            name="Sensors"
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position',
                zaxis=dict(range=[0, z_max])
            ),
            title="3D HDOP Distribution on Bottom Plane of Box"
        )

        fig.show()


    # Get DOP values for a specific point
    def get_dop_values_at_point(self, point):
        """
        Get PDOP, HDOP, and VDOP values for a specific point.
        Args:
            point (np.ndarray): A 3D point [x, y, z].
        Returns:
            tuple: PDOP, HDOP, and VDOP values.
        """
        G = self.calculate_geometry_matrix(point)
        pdop, hdop, vdop = self.calculate_dop_values(G)
        return hdop
        
    def plot_HDOP_Flatgrid(self, dot_size=100): 
        """
        Plot the HDOP values on a 2D plane (z = 0) using a 2D scatter plot.
        """
        x_min, x_max, y_min, y_max, _, _ = self.bounds
        x_vals = np.arange(x_min, x_max + self.grid_resolution, self.grid_resolution)
        y_vals = np.arange(y_min, y_max + self.grid_resolution, self.grid_resolution)
        z_fixed = 0
        hdop_grid = np.zeros((len(x_vals), len(y_vals)))
        np.zeros((len(x_vals), len(y_vals)))

        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                point = np.array([x, y, z_fixed])
                G = self.calculate_geometry_matrix(point)
                _, hdop, _ = self.calculate_dop_values(G)
                hdop_grid[i, j] = hdop

        x_flat = np.repeat(x_vals, len(y_vals))
        y_flat = np.tile(y_vals, len(x_vals))
        z_flat = np.full_like(x_flat, z_fixed)
        hdop_flat = hdop_grid.ravel()

        # cutoff HDOP values of great elevation for color mapping (scale to fit max height of 10)
        
        color_Hdop = np.clip(hdop_flat, 0, self.max_color)
        color_val = 1 / (color_Hdop**2 + 1e-6) if self.invert_color else color_Hdop

        plt.figure(figsize=(10, 8))
        
        plt.scatter(x_flat, y_flat, c=color_val, cmap=self.color_style , s=dot_size)
        plt.colorbar(label='HDOP')
        plt.title("HDOP distriubutions")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid(True)

        plt.show()

    def Moving_sensor(self, movement_matrix, steps, rows=2):
        """
        Plot HDOP distributions with a moving sensor as a set of subplots.
        Args:
            movement_matrix (np.ndarray): Matrix specifying the movement of sensors at each step.
            steps (int): Number of steps to simulate.
            rows (int): Number of rows for the subplots. Default is 2.
        """
        sensor_pos = self.sensors.copy()

        # Calculate the number of columns based on the number of steps and rows
        cols = (steps + rows - 1) // rows  # Ceiling division to ensure all steps fit

        # Create subplots for each step
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
        fig.suptitle("HDOP Distributions with Moving Sensor", fontsize=16)
        fig.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust space between subplots

        # Flatten the axes array for easier indexing
        axs = axs.flatten()

        for i in range(steps):
            # Update sensor positions
            self.sensors += movement_matrix

            # Get HDOP map for the current sensor configuration
            x_flat, y_flat, color_val, hdop, x_max, x_min, y_max, y_min, _ = self.get_DOP_map(sensor_pos)
            # point at lowest value not nan
            min_index = np.nanargmin(hdop)
            print(f"Step {i + 1} - Min HDOP: {hdop[min_index]} at ({x_flat[min_index]}, {y_flat[min_index]})")
            min_x = x_flat[min_index]
            min_y = y_flat[min_index]



            # Plot on the current subplot
            scatter = axs[i].scatter(x_flat, y_flat, c=color_val, cmap=self.color_style, s=100)
            # Plot sensor positions
            axs[i].scatter(min_x, min_y, color='green', s=200, label='Min HDOP Point')
            axs[i].scatter(self.sensors[:, 0], self.sensors[:, 1], color='red', s=200, label='Sensors')
            axs[i].set_title(f"Step {i + 1}")
            axs[i].set_xlabel("X Position")
            axs[i].set_ylabel("Y Position")
            axs[i].set_xlim(x_min, x_max)
            axs[i].set_ylim(y_min, y_max)
            axs[i].grid(True)

        # Hide any unused subplots
        for j in range(steps, len(axs)):
            axs[j].axis('off')

        # Add a single colorbar for the entire figure
        cbar = fig.colorbar(scatter, ax=axs, orientation='vertical', shrink=0.8, pad=0.1)
        cbar.set_label("HDOP")
        axs[0].legend(loc='upper right')

        plt.show()

    def get_DOP_map(self, sensor):
        
        x_min, x_max, y_min, y_max, _, _ = self.bounds
        x_vals = np.arange(x_min, x_max + self.grid_resolution, self.grid_resolution)
        y_vals = np.arange(y_min, y_max + self.grid_resolution, self.grid_resolution)
        z_fixed = 0
        hdop_grid = np.zeros((len(x_vals), len(y_vals)))
        np.zeros((len(x_vals), len(y_vals)))


        for j, x in enumerate(x_vals):
            for k, y in enumerate(y_vals):
                point = np.array([x, y, z_fixed])
                G = self.calculate_geometry_matrix(point)
                _, hdop, _ = self.calculate_dop_values(G)
                hdop_grid[j, k] = hdop

        x_flat = np.repeat(x_vals, len(y_vals))
        y_flat = np.tile(y_vals, len(x_vals))
        z_flat = np.full_like(x_flat, z_fixed)
        hdop_flat = hdop_grid.ravel()

        # cutoff HDOP values of great elevation for color mapping (scale to fit max height of 10)
        
        color_Hdop = np.clip(hdop_flat, 0, self.max_color)
        color_val = 1 / (color_Hdop**2 + 1e-6) if self.invert_color else color_Hdop

        return x_flat, y_flat, color_val, hdop_flat, x_max, x_min, y_max, y_min, z_fixed




