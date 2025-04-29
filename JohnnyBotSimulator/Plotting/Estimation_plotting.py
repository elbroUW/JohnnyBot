import matplotlib.pyplot as plt
import numpy as np
import sys
import os


class EstimationPlotting:
    def __init__(self) -> None:
        self.true_position = None
        self.sensor_positions = None
        self.noisy_positions = None
        self.measurement_number = None
        self.num_tests = 1
    

    #plot X over number of measurments with true measument 
    def plot_measurements_X(self):
        fig, ax = plt.subplots()
        ax.set_title("Measurements")
        ax.set_xlabel("Measurement Number")
        ax.set_ylabel("X Position")

       
        self.num_tests = len(self.true_position)
        
        # Plot noisy positions
        for i in range(self.num_tests):
           
            true_positions = np.tile(self.true_position[i][0], (self.measurement_number.shape[0], 1))

            ax.plot(self.measurement_number, self.noisy_positions[i][0]- true_positions.T[0], label=f'Noisy Position for ({self.true_position[i][0]}, {self.true_position[i][1]})')
        
        ax.legend()
        plt.show()
    #plot Y over number of measurments with true measument 
    def plot_measurements_Y(self):
        fig, ax = plt.subplots()
        ax.set_title("Measurements")
        ax.set_xlabel("Measurement Number")
        ax.set_ylabel("Y Position")

        true_positions = np.tile(self.true_position, (self.measurement_number.shape[0], 1))
        

        
        # Plot noisy positions
        for i in range(self.num_tests):
            true_positions = np.tile(self.true_position[i][1], (self.measurement_number.shape[0], 1))
            
            ax.plot(self.measurement_number, self.noisy_positions[i][1] - true_positions[0], label=f'Noisy Position for ({self.true_position[i][0]}, {self.true_position[i][1]})')
        
        
        ax.legend()
        plt.show()

  
    