# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:01:33 2024

@author: Isabella Askjær Gaarde Lorenzen
"""

# ================================================= #
#                Importing packages                 #
# ================================================= #

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import pygimli as pg
import os
import time


# ================================================= #
#                   Introduction                    #
# ================================================= #

'''
This scripts conducts the full reprocessing of the raw ip data for a single 
data point, and includes plots, to assess the process. 

This script of reprocessing the raw ip data is based on the article:
"Doubling the spectrum of time-domain induced polarization by harmonic de-noising,
drift correction, spike removal, tapered gating and data uncertainty estimation" (2016) 
written by Olson, P.I. et al.    

This script is written with assistance of ChatGPT. The script is designed to process
full waveform data with 100% duty cycle, recorded with the multi gradient protocol.

Additionally this code is written for 1000 Hz sampling rate (information can be found in
project file using Terrameter Toolbox). If a dataset with a different sampling rate is to be
processed adjustments to this script should be made accordinly. 
'''


# ================================================= #
#                 Input parameters                  #
# ================================================= #

# Choose an electrode seperation 
electrode_seperation = 5

# Adjust this multiplier to see the impact
current_multiplier = 2

# Enter an approximate number of data points. This will define how many iterations
# the main loop in the processing section will undergo, and should cover the number
# of input data files. 
num_datapoints = 2000

# Choose a delay time (e.g. 10 ms) (removes datapoints from identified switch spikes to x ms on)
delay_time = 10

# Please enter the acqusition time in ms (used in stacking). 
acq_time = 300

# Choose the appropriate sampling rate
fs = 1000 # 1000 Hz for 1 ms sampling

# Choose a value for the maximum number of iterations for the cole cole drift models. 
max_iterations = 2000 

# Choose a base for the increasing log. of the gate sizes for the gating of IP-data (e.g. 1.3)
base = 1.3 # Base for the logarithmic increase of the gating of the stacked full waveform data    

# Import data point id data (text file from Terrameter LS Toolbox with MeasID, DPID, Channel)
DPID_data = pd.read_csv('C:/Users/Isabe/Desktop/Regate IP-data/Ilulialik/DPID.txt', sep=',', skipinitialspace=True)

# Importing the .dat file with only resistivity data, to attach the new IP-data 
DAT_file_ResOnly = pg.physics.ert.load("C:/Users/Isabe/Desktop/Regate IP-data/Ilulialik/Ilulialik_ResOnly.dat")


# Iterating import of the full wave data (Base path and file name template)
base_path = 'C:/Users/Isabe/Desktop/Regate IP-data/Full wave data/0002-'
file_template = base_path + '{:06d}.txt'



# ================================================= #
#                   Preparing data                  #
# ================================================= #

# Display settings of row/column in the kernel (preference for viewing data in the kernel)
pd.set_option("display.max_rows", 15)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


# ================================================= #
#       Processing each data point subsequently     #
# ================================================= #

# Initiate a list to store all processed gated IP data if the code has run through all data points.
IP_values = []

# Initiate a list to store the DPID in the same order IP_value is stored. 
DPID_list = []

# Initiate a loop to iterate over each full waveform measurements downloaded from Terrameter Toolbox
for file_number in range(1, num_datapoints):
    file_path = file_template.format(file_number)
    if not os.path.exists(file_path):
        continue
    
    if file_number not in DPID_data['MeasID'].values:
        continue
    
    print(f"Computing regating of data points for measure ID {file_number}")
    
    # Importing the current full waveform data
    data = pd.read_csv(file_path, sep='\t', skipinitialspace=True)

    
    # Multiply the selected columns by 1000 and assign the result back to the original DataFrame
    data.iloc[:, 1:-1] *= 1000 # Converting from V and A to mV and mA
    
    # Rename columns with appropriate units
    for col in data.columns:
        if 'V' in col:
            data.rename(columns={col: col.replace('[V]', '[mV]')}, inplace=True)
        elif 'A' in col:
            data.rename(columns={col: col.replace('[A]', '[mA]')}, inplace=True)

    # Processing each of the data point in the measurement file (usually containing 
    # 1-4 data point)
    # Counting the number of columns that start with 'Urx' (the number of data points in the
    # data file.)
    urx_columns_count = len([col for col in data.columns if col.startswith('Urx')])    
    for Q in range(1, urx_columns_count + 1):

        # Slicing row containing current data point id (DPID)
        # file_number contains the measurement ID and Q the channel number, which can identify the DPID in the DPID textfile. 
        DPID_row = DPID_data[(DPID_data['MeasID'] == file_number) & (DPID_data['Channel'] == Q)]
        
        # If the DPID exist the data point will be processed. 
        if not DPID_row.empty:
            # Retrieving the current DPID
            DPID_current = DPID_row['DPID'].values[0]
            DPID_list.append(DPID_current)
        
        
        # Select the current voltage data point in this measurement (usually containing 1-4 data points)
        voltage_column = f'Urx({Q})[mV]'
    
        # ================================================= #
        #                   Spike removal  
        # ================================================= #
        
        # Identify and filter switch spikes
        
        # Extract relevant columns
        time = data['Time[ms]']
        voltage = data[voltage_column]
        current = data['Itx[mA]']
        
        # Calculate the difference in voltage and current to identify spikes
        current_diff = current.diff()
        
        # Function to identify spikes with adjustable multiplier
        def identify_spikes_current_only(current_diff, current_multiplier=3):
            current_threshold = current_diff.std() * current_multiplier
            current_spike_indices = current_diff[abs(current_diff) > current_threshold].index
            return current_spike_indices
        
        # Identify the spikes using the chosen multiplier
        spike_indices = identify_spikes_current_only(current_diff, current_multiplier)
        
        
        # Plotting the detected spikes before cleaning
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('Potential [mV]')
        scatter1 = ax1.scatter(time, voltage, label=f'Potential for DPID {DPID_current} [mV]', color='red', s=1, zorder=3)
        ax1.tick_params(axis='y')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Current [mA]', color='black')
        scatter2 = ax2.scatter(time, current, label='Current [mA]', color='black', s=1, zorder=1)
        ax2.tick_params(axis='y', labelcolor='black')
        scatter3 = ax1.scatter(time[spike_indices], voltage[spike_indices], color='blue', label='Flagged Switch-Spike', zorder=5)
        # Combine all handles and labels
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        fig.tight_layout()
        ax1.legend(handles, labels, loc='lower left')
        plt.title('Full waveform data with the potential and current with flagged switch-spikes')
        plt.grid(True)
        plt.show()
        
        
        # Output the spike times and values
        spike_data = data.loc[spike_indices, ['Time[ms]', voltage_column, 'Itx[mA]']]
        
        # Make a copy of the data
        despiked_switch_spikes_data = data.copy()
        
        # List to hold indices of data points removed
        removed_indices = []

        # Remove the data points around each detected switch-spike within the delay time
        # This loop flags and removes data points a full delay time and a half delay time around a flagged switch-spike. 
        for spike_index in spike_indices:
            spike_time = time[spike_index]
            half_delay_time = delay_time / 2
        
            # Indices to remove
            to_remove = despiked_switch_spikes_data[
                (despiked_switch_spikes_data['Time[ms]'] >= (spike_time - half_delay_time)) & 
                (despiked_switch_spikes_data['Time[ms]'] <= spike_time + delay_time)
            ].index
            removed_indices.extend(to_remove)
            despiked_switch_spikes_data = despiked_switch_spikes_data.drop(to_remove)
        
        # Create a list of data indices for each section between the spikes (used in trend removal section)
        all_indices = set(data.index)
        removed_indices = set(removed_indices)
        remaining_indices = sorted(all_indices - removed_indices)
        
        # Group the remaining indices into sections between the removed spikes, to define the pulses.
        sections = []
        current_section = []
        
        for i in remaining_indices:
            if not current_section or i == current_section[-1] + 1:
                current_section.append(i)
            else:
                sections.append(current_section)
                current_section = [i]
        if current_section:
            sections.append(current_section)
        
        # Re-extract the relevant columns after removing spikes
        time = despiked_switch_spikes_data['Time[ms]']
        voltage = despiked_switch_spikes_data[voltage_column]
        current = despiked_switch_spikes_data['Itx[mA]']


        # =================================================
        #      Identify and filter de-spike spikes 
        # =================================================
        
        # Convert 'Urx(1)[mV]' to volts
        data_Volt = data[voltage_column] / 1000  # Convert mV to V
    
    
        # 1) Compute the difference of a the dataframe element compared with the element in the previous row (default is element in previous row).
        u2_n = data_Volt.diff()
        
    
        # 2) Applying a non-linear energy operator 
    
        u3_n = []
        for i in range(1, len(u2_n) - 1):
            u3_i = abs((u2_n[i]**2) - (u2_n[i-1] * u2_n[i+1]))
            u3_n.append(u3_i)
    
        # Convert u3_n to a pandas Series
        u3_n = pd.Series(u3_n)
    
    
        # 3) Downsample for maximum value with 20 samples
    
        # Define the downsampling factor
        downsampling_factor = 20
    
        # Downsample u3_n by taking the maximum value within every 20 samples
        downsampled_u3_n = u3_n.groupby(u3_n.index // downsampling_factor).max()
    
    
        # 4) Applying Hampel filter
    
        Hampel_check_spikes = [] # Initiate a list to check which and how many data points have been filtered
        def hampel_filter(data, window_size=4, n_sigma=3):
    
            filtered_data = np.copy(data)  # Copy input data to avoid modifying original
    
            for i in range(len(data)):
                window = data[max(0, i - window_size//2): min(len(data), i + window_size//2 + 1)]  # Define window
                median = np.median(window)  # Compute median of window
                mad = np.mean(np.abs(window - np.mean(window)))  # Compute mean absolute deviation (MAD)
                
                # Replace sample value with median if it differs more than n_sigma * STDs from median
                if np.abs(data[i] - median) > n_sigma * mad:
                    filtered_data[i] = median
                    Hampel_check_spikes.append(i)
    
            return filtered_data
    
        u3_n_hampel_filtered = hampel_filter(downsampled_u3_n, window_size=4, n_sigma=3)
    
    
        # 5) The output from step 4 is interpolated with linear interpolation for each sample index in u3.
    
        # Define the indices of the original u3_n
        original_indices = np.arange(len(u3_n))
    
        # Interpolate the filtered data to the original indices
        interpolated_u3_n_hampel_filtered = interp1d(
            np.linspace(0, len(downsampled_u3_n) - 1, len(downsampled_u3_n)), 
            u3_n_hampel_filtered, 
            kind='linear'
        )(np.linspace(0, len(downsampled_u3_n) - 1, len(u3_n)))
    

        # 6) De-spike the data for identified de-spike spikes
    
        # This function computes threshold values to identify spikes based on the smoothed interpolated data
        Threshold = [] # Initiate a list of thesholds for plotting
        def hampel_threshold(smooth_data, window_size=4, n_sigma=3):
    
            for i in range(len(smooth_data)):
                window = smooth_data[max(0, i - window_size//2): min(len(smooth_data), i + window_size//2 + 1)]  # Define window
                # median = np.median(window)  # Compute median of window
                mad = np.mean(np.abs(window - np.mean(window)))  # Compute mean absolute deviation (MAD)
                
                Threshold.append(n_sigma * mad)
                
            return Threshold
    
        Threshold = hampel_threshold(interpolated_u3_n_hampel_filtered, window_size=4, n_sigma=3)
    
    
        # This function identify spikes in the original data from step 2 based on threshold values computed from the smoothed interpolated data
        def hampel_identify(original_data, window_size=4, n_sigma=3):
            Hampel_identify_spikes = []  # Local list for identified spikes
    
            for i in range(len(original_data)):
                window = original_data[max(0, i - window_size//2): min(len(original_data), i + window_size//2 + 1)]  # Define window
                median = np.median(window)  # Compute median of window     
                
                # Identify spikes based on threshold
                if np.abs(original_data[i] - median) > Threshold[i]:
                    Hampel_identify_spikes.append(i)
    
            return Hampel_identify_spikes
    
        spike_indices = hampel_identify(u3_n, window_size=4, n_sigma=3)
    
    
        # This function replaces the data points flagged as spikes with the window median of non-spike data points
        # to the full waveform data.
        def hampel_replace(fullwaveform_data, spikes_indices, window_size=4, n_sigma=3):
            filtered_data = np.copy(fullwaveform_data)  # Copy input data to avoid modifying original
    
            for i in range(len(fullwaveform_data)):
                # Define window range
                start = max(0, i - window_size // 2)
                end = min(len(fullwaveform_data), i + window_size // 2 + 1)
                window = fullwaveform_data[start:end]
                
                # Identify non-spike values in the window
                non_spike_window = [window[j] for j in range(len(window)) if start + j not in spikes_indices]
                
                if len(non_spike_window) > 0:
                    median = np.median(non_spike_window)  # Compute median of non-spike values in window
                else:
                    median = np.median(window)  # Fallback to median of all values if no non-spike values
    
                # Check if the current index is in spikes_indices
                if i in spikes_indices:
                    # Replace the spike with the median of non-spike values in the window
                    filtered_data[i] = median
                    #print(f"Spike at index {i} replaced with median {median}")
            
            return filtered_data
    
        # Apply the Hampel filter
        despiked_data = pd.DataFrame(hampel_replace(data[voltage_column].values, spike_indices))
    

        # Defining the dataset for applying despiking for both despike spikes and switch spikes. 
        despiked_all_data = despiked_data.copy()
        time = data['Time[ms]']
    
        # Remove the data points corresponding to switch spikes from the despiked data
        despiked_all_data.drop(index=removed_indices, inplace=True)
        time.drop(index=removed_indices, inplace=True)
        
        # Plot the fully despiked full waveform data
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('Potential [mV]')
        plt.scatter(data['Time[ms]'], data[f'Urx({Q})[mV]'], label=f'Raw potential for DPID {DPID_current}', color='b', s=1)
        plt.scatter(time, despiked_all_data, label=f'Despiked potential for DPID {DPID_current}', color='r', s=1)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Current [mA]', color='black')
        ax2.scatter(time, current, label='Current [mV]', color='black', s=1)
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        ax1.legend(handles, labels, loc='lower left')
        plt.title('Full waveform data with potentials and currents after removing switch-spikes')
        plt.grid(True)
        plt.show()
        
        
        # ================================================= #
        #                   Trend removal                   #
        # ================================================= #
    
        data_despiked = despiked_all_data
    
    
        # To define the cole-cole parameters the last 40% the four pulses is utilized. 
        # Therefore this section will begin by retrieving the last 40% of data points for 
        # each section, and then continue by gating the data points to reduce the amout of
        # data points for the fitting of the cole-cole parameters. 
    
        # Function to get the last 40% of data points in each section
        def get_last_40_percent(sections):
            last_40_percent_sections = []
            for section in sections:
                num_points = len(section)
                last_40_percent_count = int(num_points * 0.4)
                last_40_percent_sections.append(section[-last_40_percent_count:])
            return last_40_percent_sections
    
        # Retrieve the last 40% of the data points in each section
        last_40_percent_indices = get_last_40_percent(sections)
        
        # Retrieving only the last 40% indices of the four pulses. 
        subset_indices = last_40_percent_indices[2:-1]
    
    
        # Adding the despiked data to the data dataframe as input to the for-loop below. 
        # Add a new column 'Despiked' to 'data' and initialize with NaN
        data[f'Despiked_Urx({Q})[mV]'] = np.nan
    
        # Update the 'Despiked' column with values from 'data_despiked' using their indices
        data.loc[data_despiked.index, f'Despiked_Urx({Q})[mV]'] = data_despiked.values
    
    
        # Gated subset of the full waveform data. The data density of the subset is reduced 
        # by considering a gating schemes of 20 ms for all gates in the subset data. 
        # Width of the window is set to 1 ms (for 1000 Hz) (N_fo_samples = 20 samples for 20 ms)
        u_subsets_gated = []
        u_subsets_gated_indices = []
        for k in range(0, len(subset_indices)):
            N_fo_samples = 20
            interval_range = len(subset_indices[k]) // N_fo_samples 
            
            gate_averages = []  # List to store the average values for each 20 ms interval
            middle_indices = []  # List to store indices in the middle of the interval for each k
            
            # Compute the gated value 
            # Loop through the desired number of intervals
            for interval in range(interval_range):
                gate_sum = 0  # Reset the sum for each interval
                interval_indices = []
                
                # j is the data point in a subsection. This loop considers a gate of 20 ms, sum up all of the
                # values on the loop and later averages it with N_fo_samples. 
                for j in range(1, N_fo_samples + 1):
                    gate_sum += (data[f'Despiked_Urx({Q})[mV]'].iloc[subset_indices[k][0] + interval * N_fo_samples + j - 1])
                    
                    sample_index = subset_indices[k][0] + interval * N_fo_samples + j - 1
                    interval_indices.append(sample_index)
                    
                # Calculate the average for the current interval
                gate_avg = gate_sum / N_fo_samples
            
                # Append the average value to the list of gate averages
                gate_averages.append(gate_avg)
                
                # Add the indices in the middle of the interval to the middle_indices list
                middle_index = len(interval_indices) // 2
                middle_indices.append(interval_indices[middle_index])
                
            u_subsets_gated.append(gate_averages)
            u_subsets_gated_indices.append(middle_indices)
        
        # This defines the indices and data of all of the gated subsets which will
        # be used to fit the cole cole parameters. 
        CC_indices = u_subsets_gated_indices
        CC_data = u_subsets_gated
    
        
        # Defining the cole cole drift model of which the parameters m0, tau, c, d1 and d2
        # will be fitted. This model considers two cole cole drift models with the same m0, tau
        # and c but different offsets d1 and d2 for the positive half-periods and the negative
        # half-periods, respectively. 
        def cole_cole_model(n, m0, tau, c, d1, d2, fs, idx):
            model = np.zeros_like(n, dtype=np.float64)
            
            # A tolerance is defined of which the optimization algorithem stops
            # when convergence is be below the threshold. 
            tolerance = 1e-3 
            previous_model = np.zeros_like(n, dtype=np.float64)
            
            # The iteration number j should go from 0 to infinity, but to reduce
            # computation effords the maximum number of iterations is set to a finite number, 
            # and breaks if the compuation lead to an infinate or NaN result. 
            for j in range(0, max_iterations):  # Use a reasonable number of terms for approximation
                term = (-1)**j * (n / (tau * fs))**(j * c) / gamma(1 + (j * c))
                model += term
                if np.all(np.abs((model - previous_model) / (previous_model + np.finfo(float).eps)) < tolerance):
                    break
                previous_model = np.copy(model)
            
            d = np.where((idx == 0) | (idx == 2), d1, d2)  # Choose d1 for 0th and 2nd subsets, and d2 for 1st and 3rd subsets.
            return m0 * model + d
    
        # This flattes the indices and data of all of the gated subsets which will
        # be used to fit the cole cole parameters. 
        all_indices = np.concatenate(CC_indices)
        all_data = np.concatenate(CC_data)
        
        # Creating an array indicating which subset each data point belongs to.
        idx = np.concatenate([np.full_like(CC_indices[i], i) for i in range(len(CC_indices))])
    
        # Defining the function to fit the cole cole parameters from the cole cole drift model
        # to the gated subsections (last 40% of the half periods with a 20ms gating scheme).
        def fit_cole_cole_model(n, u_combined, fs, idx):
            # Guessing initial parameters
            initial_params = [np.median(all_data), np.mean(np.diff(time)), 0.5, 0, 0]  # Initial guesses for m0, tau, c, d1, d2
            params, _ = curve_fit(lambda n, m0, tau, c, d1, d2: cole_cole_model(n, m0, tau, c, d1, d2, fs, idx),
                                  n, u_combined, p0=initial_params, maxfev=100000)
            return params
    
        # Defining the fitted cole cole parameters.
        fitted_params = fit_cole_cole_model(all_indices, all_data, fs, idx)
        print(f"Fitted parameters:\n m0: {fitted_params[0]}\n tau: {fitted_params[1]}\n c: {fitted_params[2]}\n d1: {fitted_params[3]}\n d2: {fitted_params[4]}")
        m0, tau, c, d1, d2 = fitted_params
    
        # Defining the average offset d_avg
        d_avg = (d1 + d2) / 2
    
        # Defining a cole-cole drift model to compute the drift, which takes the
        # fitted cole cole parameters as input, with only one offset value 
        # for the averaged offset value d_avg. 
        def cole_cole_model2(n, m0, tau, c, d, fs):
            model = np.zeros_like(n, dtype=np.float64)
            
            # A tolerance is defined of which the optimization algorithem stops
            # when convergence is be below the threshold. 
            tolerance = 1e-3  
            previous_model = np.zeros_like(n, dtype=np.float64)
            
            # The iteration number j should go from 0 to infinity, but to reduce
            # computation effords the maximum number of iterations is set to a finite number, 
            # and breaks if the compuation lead to an infinate or NaN result.
            for j in range(0, max_iterations):  # Use a reasonable number of terms for approximation
                term = (-1)**j * (n / (tau * fs))**(j * c) / gamma(1 + (j * c))
                model += term
                if np.all(np.abs((model - previous_model) / (previous_model + np.finfo(float).eps)) < tolerance):
                    break
                previous_model = np.copy(model)
            return m0 * model + d
  
        
        # Computing the drift for each section in the full waveform data. 
        section_indices = sections
    
        u_drift_lists = []
        for section in section_indices:
            u_drift_list = cole_cole_model2(np.array(section), m0, tau, c, d_avg, fs)
            u_drift_lists.append(u_drift_list)
    
        # Subtracting the drift from the measured value (u_processed = u_measured - u_drift)
        u_processed_subsets = []
        for section, drift_list in zip(section_indices, u_drift_lists):
            u_processed_subset = data[f'Despiked_Urx({Q})[mV]'].iloc[section].values - drift_list
            u_processed_subsets.append(u_processed_subset)
    
        # Flatten the section_indices and u_processed_subsets lists
        flat_indices = [index for sublist in section_indices for index in sublist]
        flat_data = [item for sublist in u_processed_subsets for item in sublist]
    
        # Create a new column and initialize it with NaN
        data[f'Processed_Urx({Q})[mV]'] = np.nan
    
        # Assign the flattened data to the corresponding indices in the DataFrame
        data.loc[flat_indices, f'Processed_Urx({Q})[mV]'] = flat_data
            
        # Plotting the trend removal and despiked full waveform data
        plt.figure(figsize=(10, 6))
        plt.scatter(data['Time[ms]'], data[f'Processed_Urx({Q})[mV]'], label=f'Processed full waveform data for DPID {DPID_current}', color='r', s=1)
        plt.title('Full waveform data corrected for drift and DC offset')
        plt.xlabel('Time [ms]')
        plt.ylabel('Potential [mV]')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.show()
        
        
        # ================================================= #
        #          Stacking of full waveform data           #
        # ================================================= #
            
        # Defining the sections used for stacking. This is the last three half-periods. 
        # (two negative and one positive half-period).
        S_IP = section_indices[3:-1]
    
        # Defining the number of pulses (number of half-periods).
        N_pulses = len(S_IP)
        # Defining the number of data points used in stacking, as the number
        # of data points of the shortest pulse. 
        K_data_points = min([len(sublist) for sublist in S_IP])
                                        
        # Conducting the stacking of the processed data of the defined pulses                                                 
        u_IP_stacked = []
        for k in range(1, K_data_points + 1):  
            # j = pulses
            # k = data point increment
            
            # Initialize u_subset to accumulate the sum of values for each pulse
            u_IP_stacked_sum = 0
            
            for j in range(1, N_pulses + 1):
                u_IP_stacked_sum += (-1)**(j+1) * (data[f'Processed_Urx({Q})[mV]'].iloc[k + S_IP[j-1][0] - 1])           
                
            u_IP_stacked_value = u_IP_stacked_sum / N_pulses
            
            u_IP_stacked.append(u_IP_stacked_value)

    
        # Computes an average of the voltage during the acquisition time of the resistivity measurement,
        # as the last e.g. 300 ms of the stacked curve, if the acquisition time is 0.3 s.
        # This is used to correct for DC-offset. 
        u_IP_stacked_acq_time = u_IP_stacked[-acq_time:]
        N_samples = acq_time
        DC_gated_sum = 0
        for i in range(0, len(u_IP_stacked_acq_time)):
            DC_gated_sum += u_IP_stacked_acq_time[i]
        DC_gated = DC_gated_sum / N_samples
        
        
        # Correcting the stacked decay curve for DC-offset. 
        u_IP_corrected = ((u_IP_stacked / (DC_gated))*(-1) + 1)
        u_IP_corrected = (u_IP_corrected*1000) # in mv/V
    
        
        plt.figure(figsize=[10, 6])
        plt.plot(u_IP_corrected, marker='.', linestyle = '-', color = 'r')
        plt.title(f'Stacked full waveform IP -data - DPID {DPID_current}')
        plt.xlabel('Time [ms]')
        plt.ylabel('IP values [mV/V]')
        plt.grid(True)
        plt.show()
        
                
        # ================================================= #
        #               Gating of full waveform             #
        # ================================================= #
    
        # Defining the IP-data for gating
        IP_FW = pd.DataFrame(u_IP_corrected) 
    
        # This divides the indices of the stacked full waveform decay curve into gates of 
        # a logarithmic gating scheme, that is a multiplier of 20 ms (for 1 s sampling rate)
        initial_size = 20
        n = len(IP_FW)
        gates = []
        current_index = 0
        gate_number = 0
        while current_index < n:
            # Calculate the next gate size using logarithmic increase
            next_size = int(initial_size * (base ** gate_number))
            
            # Ensure the gate size is a multiple of 20
            next_size = ((next_size + 19) // 20) * 20  # Round up to the nearest multiple of 20
            
            # Ensure we do not go out of the bounds of the DataFrame
            end_index = min(current_index + next_size, n)
            
            # Append the gate indices
            gates.append(list(range(current_index, end_index)))
            
            # Update the current index and gate number
            current_index = end_index
            gate_number += 1
    
    
        # Computing the ip-value for each gate.
        gate_values = []
        IP_times = []
        for m in range(0,len(gates)):
            N_samples = len(gates[m])
            gate_sum = 0 
            for i in range(1, N_samples+1):
                gate_sum += IP_FW.iloc[i + gates[m][0] - 1, 0 ]
                
            # Calculate the average of the gate
            gate_avg = gate_sum / N_samples
            
            #Append the gated value to the list of gate values
            gate_values.append(gate_avg)
            
            mid_index = gates[m][N_samples // 2]
            IP_times.append(mid_index)
                
        IP_values.append(gate_values) 
            
        # Plot the gate values
        plt.plot(IP_times, gate_values, marker='o', linestyle='-', label=f'DPID {DPID_current}', color = 'r')
    
        # Adding labels and title
        plt.xlabel('Time [ms]')
        plt.ylabel('IP value [mV/mV]')
        plt.title(f'Gated IP curve - DPID {DPID_current}')
        plt.legend()
        plt.grid(True)
        plt.show()
        

# Creating a dataframe compining the DPID and the IP-values
ip_values_df = pd.DataFrame(IP_values)

# Rename the columns to IP#1(mV/V), IP#2(mV/V), etc.
ip_values_df.columns = [f'ip{i+1}' for i in range(ip_values_df.shape[1])]

# Add the DPID_list as the first column
final_df = pd.DataFrame({'DPID': DPID_list})

# Concatenate the two DataFrames along the columns
IP_data = pd.concat([final_df, ip_values_df], axis=1)

# Attaching coordinates to the IP_data dataframe
IP_data = pd.merge(IP_data, DPID_data[['DPID', 'A(x)', 'B(x)', 'M(x)', 'N(x)']], on='DPID', how='left')



