# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:14:42 2024

@author: Isabella Askjær Gaarde Lorenzen (s194835)
"""

# ================================================= #
#                Importing packages                 #
# ================================================= #

import pandas as pd
import matplotlib.pyplot as plt
import glob

# ================================================= #
#                   Introduction                    #
# ================================================= #

'''
This script subsequently plots the despiked full waveform data based on the results
from the script 1.1 Despike full waveform data. 

This script is written with assistance from ChatGPT

'''

# ================================================= #
#  Plotting each despiked full waveform data point  #
# ================================================= #

# Define the directory where the despiked data files are stored
despiked_data_directory = 'C:/Users/Isabe/Desktop/Regate IP-data/Ilulialik/Full wave data_despiked/'

# Get a list of all despiked data files
file_paths = glob.glob(despiked_data_directory + '*_despiked.txt')

# Function to plot each file
def plot_despiked_data(file_path):
    # Read the data
    data = pd.read_csv(file_path, sep='\t')
    
    # Plot each column separately
    for column in data.columns[1:]:  # Skip the Time column
        plt.figure(figsize=(10, 3))
        plt.scatter(data['Time[ms]'], data[column], label=f'Despiked IP-data - DPID {column}', color='r', s=1)
        
        # Set plot title and labels
        plt.title(f'Despiked full wavefprm potentialer- DPID {column}')
        plt.xlabel('Tid [ms]')
        plt.ylabel('IP-værdi [mV]')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.show()

# Plot all despiked data files
for file_path in file_paths:
    plot_despiked_data(file_path)
