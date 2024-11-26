# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:01:33 2024

@author: Isabella Askjær Gaarde Lorenzen (s194835)
"""

# ================================================= #
#                Importing packages                 #
# ================================================= #

import numpy as np
import pandas as pd
import os
import glob
import logging
from scipy.interpolate import interp1d
from multiprocessing import cpu_count, Pool

# ================================================= #
#                   Introduction                    #
# ================================================= #

'''
This script of despiking the full waveform data is based in the article:
"Doubling the spectrum of time-domain induced polarization by harmonic de-noising,
drift correction, spike removal, tapered gating and data uncertainty estimation" (2016) 
written by Olson, P.I. et al.    

This script is written with assistance of ChatGPT. The script is designed to process
full waveform data with 100% duty cycle, recorded with the multi gradient protocol.

Additionally this code is written for 1000 Hz sampling rate (information can be found in
project file using Terrameter Toolbox). If a dataset with a different sampling rate is to be
processed adjustments to this script should be made accordinly. 

The results of this script is data file of despiked full waveform data for each input data file.

'''


# ================================================= #
#                 Input parameters                  #
# ================================================= #

# Choose a multiplier of the STD of the current data, used to identify switch spikes (e.g. 2).
current_multiplier = 2

# Choose a delay time, of which data points recorded the delay time after an identified switch spike,
# will also be identified as a switch spike (e.g. 10 ms).
delay_time = 10

# Chose the number of CPU's to process multiple data files simultaniously.
# To count the number of CPU's in the processing computer import package 'from multiprocessing import cpu_count, Pool'
# and type cpu_count(). Using half of the available CPU's is recommended (cpu_count() // 2)
num_CPU = 6  

# Choose the number of files to be processed at a time (e.g. 4)
batch_size = 4

# Choose the folder location of the downloaded full waveform data retrieved from Terrameter LS Toolbox.
input_folder_location = 'C:/Users/Isabe/Desktop/Regate IP-data/Full wave data'

# Ensure that the base path corresponds to the file names (e.g. 0002)
# (Iterating import of the full wave data (Base path and file name template))
base_path = f'{input_folder_location}/0002-' 


# Choose a folder location for the despiked full waveform data. 
output_folder_location = 'C:/Users/Isabe/Desktop/Regate IP-data/Ilulialik/Full wave data_despiked' 

# Import data point ID data (text file from Terrameter LS Toolbox with MeasID, DPID, Channel)
DPID_data = pd.read_csv('C:/Users/Isabe/Desktop/Regate IP-data/Ilulialik/DPID.txt', sep=',', skipinitialspace=True)


# ================================================= #
#               Preparing processing                #
# ================================================= #

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

file_paths = glob.glob(base_path + '*.txt')

# Display settings of row/column in the kernel (preference for viewing data in the kernel)
pd.set_option("display.max_rows", 15)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


# ================================================= #
#            Defining despiking function            #
# ================================================= #

# Function to identify switch spikes using the current data.
# Data points deviating a multiplier of the STD is flagged as a switch spikes. 
def identify_spikes_current_only(current_diff, multiplier):
    mean_diff = np.mean(current_diff)
    std_diff = np.std(current_diff)
    threshold = mean_diff + multiplier * std_diff
    spikes = current_diff[(current_diff > threshold) | (current_diff < -threshold)].index
    return spikes


# Defining the task function conducting the despiking of de-spike spikes and switch spikes. 
def process_file(file_path):
    try:
        # Defining the measurement ID (MeasID) based in the input file 
        # (Downloaded full waveform data from Terrameter Toolbox).
        file_number = int(file_path.split('-')[-1].split('.')[0])

        # Checking if the file_number is in DPID_data
        if file_number not in DPID_data['MeasID'].values:
            logging.info(f"File {file_path} not in DPID_data MeasID values")
            return None

        logging.info(f"Processing file: {file_path}")

        # Importing full wave data. 
        data = pd.read_csv(file_path, sep='\t', skipinitialspace=True)
        
        # Converting the data from V and A to mV and mA
        data.iloc[:, 1:-1] *= 1000  
        
        # Renaming the data columns with appropriate units
        for col in data.columns:
            if 'V' in col:
                data.rename(columns={col: col.replace('[V]', '[mV]')}, inplace=True)
            elif 'A' in col:
                data.rename(columns={col: col.replace('[A]', '[mA]')}, inplace=True)
        

        # Counting the number of columns that start with 'Urx'
        urx_columns_count = len([col for col in data.columns if col.startswith('Urx')])
        
        results = {}
        
        # Processing each of the data point in the measurement file (usually containing 
        # 1-4 data point). 
        for Q in range(1, urx_columns_count + 1):
            # Slicing the row containing the current data point id (DPID)
            DPID_row = DPID_data[(DPID_data['MeasID'] == file_number) & (DPID_data['Channel'] == Q)]
            
            # If the data point exist the data point will be processed. 
            if not DPID_row.empty:
                # Retrieving the current DPID
                DPID_current = DPID_row['DPID'].values[0]
            
                # Selecting the current voltage data.
                voltage_column = f'Urx({Q})[mV]'
            
                # Extract relevant columns
                time = data['Time[ms]']
                current = data['Itx[mA]']
                
                            
                # ================================================= #
                #              Smoothing despike-spikes             #
                # ================================================= #
                
                # Converting the voltage data from mV to V. 
                data_Volt = data[voltage_column] / 1000  # Convert mV to V
                
                # 1) Computing the difference of the dataframe elements compared with 
                # the element in the previous row.
                u2_n = data_Volt.diff()
                
                
                # 2) Applying a non-linear energy operator 
                u3_n = []
                for i in range(1, len(u2_n) - 1):
                    u3_i = abs((u2_n[i]**2) - (u2_n[i-1] * u2_n[i+1]))
                    u3_n.append(u3_i)
                u3_n = pd.Series(u3_n)
                
                
                # 3) Downsample for maximum value with 20 samples
                downsampling_factor = 20
                downsampled_u3_n = u3_n.groupby(u3_n.index // downsampling_factor).max()
                
                
                # 4) Applying Hampel filter
                Hampel_check_spikes = [] 
                def hampel_filter(data, window_size=4, n_sigma=3):
                    filtered_data = np.copy(data)
                    for i in range(len(data)):
                        window = data[max(0, i - window_size//2): min(len(data), i + window_size//2 + 1)]
                        median = np.median(window)
                        mad = np.mean(np.abs(window - np.mean(window)))
                        if np.abs(data[i] - median) > n_sigma * mad:
                            filtered_data[i] = median
                            Hampel_check_spikes.append(i)
                    return filtered_data

                u3_n_hampel_filtered = hampel_filter(downsampled_u3_n, window_size=4, n_sigma=3)
                
                
                # 5) The output from step 4 is interpolated with linear interpolation for each sample index in u3.
                interpolated_u3_n_hampel_filtered = interp1d(
                    np.linspace(0, len(downsampled_u3_n) - 1, len(downsampled_u3_n)), 
                    u3_n_hampel_filtered, 
                    kind='linear'
                )(np.linspace(0, len(downsampled_u3_n) - 1, len(u3_n)))
                
                
                # 6) De-spiking the data for the identified de-spike spikes
                Threshold = []
                def hampel_threshold(smooth_data, window_size=4, n_sigma=3):
                    for i in range(len(smooth_data)):
                        window = smooth_data[max(0, i - window_size//2): min(len(smooth_data), i + window_size//2 + 1)]
                        mad = np.mean(np.abs(window - np.mean(window)))
                        Threshold.append(n_sigma * mad)
                    return Threshold

                Threshold = hampel_threshold(interpolated_u3_n_hampel_filtered, window_size=4, n_sigma=3)
                
                def hampel_identify(original_data, window_size=4, n_sigma=3):
                    Hampel_identify_spikes = []
                    for i in range(len(original_data)):
                        window = original_data[max(0, i - window_size//2): min(len(original_data), i + window_size//2 + 1)]
                        median = np.median(window)
                        if np.abs(original_data[i] - median) > Threshold[i]:
                            Hampel_identify_spikes.append(i)
                    return Hampel_identify_spikes

                despike_spike_indices = hampel_identify(u3_n, window_size=4, n_sigma=3)
                
                # This function replaces the data points flagged as spikes with the window median of non-spike data points
                # to the full waveform data.
                def hampel_replace(fullwaveform_data, spikes_indices, window_size=4, n_sigma=3):
                    filtered_data = np.copy(fullwaveform_data)
                    for i in range(len(fullwaveform_data)):
                        start = max(0, i - window_size // 2)
                        end = min(len(fullwaveform_data), i + window_size // 2 + 1)
                        window = fullwaveform_data[start:end]
                        non_spike_window = [window[j] for j in range(len(window)) if start + j not in spikes_indices]
                        if len(non_spike_window) > 0:
                            median = np.median(non_spike_window)
                        else:
                            median = np.median(window)
                        if i in spikes_indices:
                            filtered_data[i] = median
                    return filtered_data

                despiked_data = pd.DataFrame(hampel_replace(data[voltage_column].values, despike_spike_indices))
                
                # Ensure the original index is preserved
                despiked_data.set_index(data.index, inplace=True)
                
                
                # ================================================= #
                #               Filtering switch spikes             #
                # ================================================= #
                                
                # Switch spike removal
                current_diff = current.diff()
                
                # Identifying switch spikes from polarization switches
                switch_spike_indices = identify_spikes_current_only(current_diff, current_multiplier)
                
                # Removing switch spikes and data points a delay time after the identified switch spikes
                # and half a delay time before the identified switch spike. 
                for spike_index in switch_spike_indices:
                    spike_time = time[spike_index]
                    half_delay_time = delay_time / 2
                    to_remove = despiked_data[
                        (data['Time[ms]'] >= (spike_time - half_delay_time)) & 
                        (data['Time[ms]'] <= spike_time + delay_time)
                    ].index
                    despiked_data.loc[to_remove] = np.nan  # Set removed values to NaN
                
                # Ensure the original index is preserved
                despiked_data.set_index(data['Time[ms]'], inplace=True)
                
                results[DPID_current] = despiked_data[0]  
        
        
        # ================================================= #
        #            Exporting the despiked data            #
        # ================================================= #
                
        # Save results to individual files
        output_file_path = f'{output_folder_location}/{os.path.basename(file_path).replace(".txt", "_despiked.txt")}'
        output_data = pd.DataFrame({'Time[ms]': time})
        for DPID_current, voltage_data in results.items():
            output_data[DPID_current] = voltage_data.values
        output_data.to_csv(output_file_path, sep='\t', index=False)
        
        return output_file_path
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None


# ======================================================== #
#    Defining function to process input files in batches   #
# ======================================================== #

def process_files_in_batches(file_paths, batch_size):
    num_cores = num_CPU   
    logging.info(f"Number of CPU cores used: {num_cores}")

    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i + batch_size]
        logging.info(f"Processing batch: {batch_paths}")

        # Create a pool of workers
        with Pool(num_cores) as pool:
            # Distribute the file paths to the workers
            results_list = pool.map(process_file, batch_paths)

        for result in results_list:
            if result is not None:
                logging.info(f"Results saved to {result}")


# ================================================= #
#              Conducting the despiking             #
# ================================================= #

if __name__ == '__main__':
    batch_size = batch_size
    logging.info(f"Starting processing with {len(file_paths)} files and batch size of {batch_size}")
    process_files_in_batches(file_paths, batch_size)
    logging.info("Processing complete")
