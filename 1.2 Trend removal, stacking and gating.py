# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:01:33 2024

@author: Isabella Askjær Gaarde Lorenzen (s194835)
"""

# ================================================= #
#                Importing packages                 #
# ================================================= #

import numpy as np
from scipy.special import gamma
import pandas as pd
from scipy.optimize import curve_fit
import os
import glob
import logging
from multiprocessing import cpu_count, Pool
import gc



# ================================================= #
#                   Introduction                    #
# ================================================= #

'''
This script of reprocessing the raw ip data is based in the article:
"Doubling the spectrum of time-domain induced polarization by harmonic de-noising,
drift correction, spike removal, tapered gating and data uncertainty estimation" (2016) 
written by Olson, P.I. et al.    

This script is written with assistance of ChatGPT. The script is designed to process
full waveform data with 100% duty cycle, recorded with the multi gradient protocol.

Additionally this code is written for 1000 Hz sampling rate (information can be found in
project file using Terrameter Toolbox). If a dataset with a different sampling rate is to be
processed adjustments to this script should be made accordinly. 

The results of this script is a text file of the reprocessed ip-data, its DPID and coordinates.
The output text file can be exported to a DAT file that can be read in Aarhus Workbench in the script
1.3 Exporting reprocessed IP-data into a DAT file. 

'''


# ================================================= #
#                 Input parameters                  #
# ================================================= #

# Choose the sampling rate in Hz (e.g. 1000 Hz for 1 ms sampling rate).
fs = 1000  

# Choose the base for the logarithmic increase of the gating scheme (e.g. 1.3).
base = 1.3  

# Please enter the electrode sepeartion from the survey. 
electrode_separation = 5

# Choose a value for the maximum number of iterations for the cole cole drift models. 
max_iterations = 5000 

# Please enter the acqusition time in ms (used in stacking). 
acq_time = 300

# Chose the number of CPU's to process multiple data files simultaniously.
# To count the number of CPU's in the processing computer import package 'from multiprocessing import cpu_count, Pool'
# and type cpu_count(). Using half of the available CPU's is recommended (cpu_count() // 2)
num_CPU = 6  

# Choose the number of files to be processed at a time (e.g. 4)
batch_size = 4


# Choose the directory containing the input text files (the despiked full waveform data).
data_directory = 'C:/Users/Isabe/Desktop/Regate IP-data/Ilulialik/Full wave data_despiked/'

# Get a list of all the relevant text files in the directory
file_list = glob.glob(os.path.join(data_directory, '*_despiked.txt'))

# Import data point id data (text file from Terrameter LS Toolbox with MeasID, DPID, Channel)
DPID_data = pd.read_csv('C:/Users/Isabe/Desktop/Regate IP-data/Ilulialik/DPID.txt', sep=',', skipinitialspace=True)

# Choose the name for the output files containing the results. 
output_name = 'gated_IP_data.txt'

# Choose the output file path for the results
output_file_path = f'C:/Users/Isabe/Desktop/Regate IP-data/Ilulialik/{output_name}'

# Choose the output file path for the log of processed data files (containing a list of the
# already processed despiked full waveform data files. 
processed_log_path = 'C:/Users/Isabe/Desktop/Regate IP-data/Ilulialik/processed_files.log'


# ================================================= #
#               Preparing processing                #
# ================================================= #

# Display settings of row/column in the kernel (preference for viewing data in the kernel)
pd.set_option("display.max_rows", 15)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================================== #
#   Defining trend removal, DC offset correction, stacking and gating function   #
# ============================================================================== #

# Variable to store the initial length of u_IP_corrected
# This is defined to truncate decay curves in the dataset that was measured for a longer
# time than the first data point, in case setting for the measuring length was changed during
# a survey. This will ensure that there is an equal number of gates for all processed data points.
initial_length = None


# Defining the task function for conducting trend removal, DC offset correction, stacking and gating
# Function to process each file
def process_file(file_path):
    # Using the global variable to store initial length
    global initial_length
    try:
        # Log the start of file processing
        logging.info(f"Starting to process file {file_path}")
        
        # Importing the current data from the measurement file file containing 1-4 data points. 
        input_data = pd.read_csv(file_path, sep='\t', skipinitialspace=True, dtype=np.float32)
        # Defining the measurement ID (MeasID) based in the input file 
        # (Downloaded full waveform data from Terrameter Toolbox).
        file_number = int(os.path.basename(file_path).split('_')[0].split('-')[1])
        
        results = []
        
        # Processing each of the data point in the measurement file (usually containing 
        # 1-4 data point)
        for Q in range(1, len(input_data.columns)):
            # Selecting the current data point from the input file.
            data_despiked = input_data.iloc[:, Q]
            
            # Defining the time data.
            time = input_data.iloc[:, 0]
            
            # Slicing the row containing the data point id (DPID).
            DPID_row = DPID_data[(DPID_data['MeasID'] == file_number) & (DPID_data['Channel'] == Q)]

            # If the DPID exist in the file the processing will be conducted on that data point. 
            if not DPID_row.empty:
                # Retrieving the current DPID 
                DPID_current = DPID_row['DPID'].values[0]
                
                # Defining the remaining indices of the data point after despiking.
                # In the despiking script the data points flagged as a switch spikes
                # where redefined as NaN, and is not removed. 
                removed_indices = data_despiked[data_despiked.isna()].index.tolist()
                all_indices = set(data_despiked.index)
                remaining_indices = sorted(all_indices - set(removed_indices))
                
                # Defining the data point sections (pulses) between the flagged switch spikes.
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
                    
                # ================================================= #
                #          Trend removal / drift correction         #
                # ================================================= #
                
                # Function to retrieve the last 40% of the data points (subsections)
                # in each section, which is used to fit the cole cole parameters. 
                def get_last_40_percent(sections):
                    last_40_percent_sections = []
                    for section in sections:
                        num_points = len(section)
                        last_40_percent_count = int(num_points * 0.4)
                        last_40_percent_sections.append(section[-last_40_percent_count:])
                    return last_40_percent_sections

                last_40_percent_indices = get_last_40_percent(sections)
                
                # Defining the subset (pulses) used to fit the cole cole parameters
                # The two positive and the two negative half periods are defined.
                subset_indices = last_40_percent_indices[2:-1]

                
                # Gating each selected subset used to fit the cole cole parameters, 
                # with a gating scheme of one gate pr. 20 ms of samples
                u_subsets_gated = []
                u_subsets_gated_indices = []
                for k in range(0, len(subset_indices)):
                    N_fo_samples = 20
                    interval_range = len(subset_indices[k]) // N_fo_samples

                    gate_averages = []
                    middle_indices = []

                    for interval in range(interval_range):
                        gate_sum = 0
                        interval_indices = []

                        for j in range(1, N_fo_samples + 1):
                            gate_sum += (data_despiked.iloc[subset_indices[k][0] + interval * N_fo_samples + j - 1])
                            sample_index = subset_indices[k][0] + interval * N_fo_samples + j - 1
                            interval_indices.append(sample_index)

                        gate_avg = gate_sum / N_fo_samples
                        gate_averages.append(gate_avg)
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
                    for j in range(0, max_iterations):
                        term = (-1)**j * (n / (tau * fs))**(j * c) / gamma(1 + (j * c))
                        model += term
                        
                        if np.all(np.abs((model - previous_model) / (previous_model + np.finfo(float).eps)) < tolerance):
                            break
                        previous_model = np.copy(model)
                    
                    # This defines the model to fit d1 for the positive half periods and d2 for the
                    # negative half periods. 
                    d = np.where((idx == 0) | (idx == 2), d1, d2)
                    
                    return m0 * model + d
                
                # This flattes the indices and data of all of the gated subsets which will
                # be used to fit the cole cole parameters. 
                all_indices = np.concatenate(CC_indices)
                all_data = np.concatenate(CC_data)
                
                
                # This lables the different subsets that are used for fitting the cole-cole parameters
                idx = np.concatenate([np.full_like(CC_indices[i], i) for i in range(len(CC_indices))])

                # Defining the function to fit the cole cole parameters from the cole cole drift model
                # to the gated subsections (last 40% of the half periods with a 20ms gating scheme).
                def fit_cole_cole_model(n, u_combined, fs, idx):
                    # Guessing initial parameters
                    initial_params = [np.median(all_data), np.mean(np.diff(time)), 0.5, 0, 0]
                    params, _ = curve_fit(lambda n, m0, tau, c, d1, d2: cole_cole_model(n, m0, tau, c, d1, d2, fs, idx),
                                          n, u_combined, p0=initial_params, maxfev=100000)
                    return params
                
                # Defining the fitted cole cole parameters.
                fitted_params = fit_cole_cole_model(all_indices, all_data, fs, idx)
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
                    for j in range(0, max_iterations):
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
                
                # Conducting the trend removal / drift correction. 
                u_processed_subsets = []
                for section, drift_list in zip(section_indices, u_drift_lists):
                    u_processed_subset = data_despiked.iloc[section].values - drift_list
                    u_processed_subsets.append(u_processed_subset)
                
                # Flattens the lists of the the corrected sections.
                flat_indices = [index for sublist in section_indices for index in sublist]
                flat_data = [item for sublist in u_processed_subsets for item in sublist]


                # ================================================= #
                #                      Stacking                     #
                # ================================================= #

                # Creating a column in the input dataframe data and appending the 
                # data that has been corrected for drift. 
                input_data['Processed_Urx(1)[mV]'] = np.nan
                input_data.loc[flat_indices, 'Processed_Urx(1)[mV]'] = flat_data
                
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
                    u_IP_stacked_sum = 0
                    for j in range(1, N_pulses + 1):
                        u_IP_stacked_sum += (-1)**(j + 1) * (input_data['Processed_Urx(1)[mV]'].iloc[k + S_IP[j - 1][0] - 1])
                    u_IP_stacked_value = u_IP_stacked_sum / N_pulses
                    u_IP_stacked.append(u_IP_stacked_value)


                # Computes an average of the voltage during the acquisition time of the resistivity measurement,
                # as the last e.g. 300 ms of the stacked curve, if the acquisition time is 0.3 s.
                # This is used to correct for DC-offset
                u_IP_stacked_acq_time = u_IP_stacked[-acq_time:]
                N_samples = acq_time
                DC_gated_sum = 0
                for i in range(0, len(u_IP_stacked_acq_time)):
                    DC_gated_sum += u_IP_stacked_acq_time[i]
                DC_gated = DC_gated_sum / N_samples


                # Correcting the stacked decay curve for DC-offset
                u_IP_corrected = ((u_IP_stacked / (DC_gated)) * (-1) + 1)
                u_IP_corrected = (u_IP_corrected*1000) # in mv/V
                
                # This truncate any datapoints where the measurement time is longer than the initial 
                # measurement time, in case measurement setting was changed during data acquisition. 
                if initial_length is None:
                    initial_length = len(u_IP_corrected)  # Set the initial length from the first file processed
                else:
                    u_IP_corrected = u_IP_corrected[:initial_length]  # Truncate to the initial length
                
                    
                # Defining the IP-data for gating   
                IP_FW = pd.DataFrame(u_IP_corrected)
                
                # This divides the indices of the stacked full waveform decay curve into gates of 
                # a logarithmic gating scheme, that is a multiplier of 20 ms (for 1 s sampling rate)
                initial_size = 20
                n = len(IP_FW)
                gates = []
                current_index = 0
                gate_number = 0
                while (current_index < n):
                    next_size = int(initial_size * (base ** gate_number))
                    next_size = ((next_size + 19) // 20) * 20
                    end_index = min(current_index + next_size, n)
                    gates.append(list(range(current_index, end_index)))
                    current_index = end_index
                    gate_number += 1
                
                # Computing the ip-value for each gate.
                IP_values = []
                IP_times = []
                for m in range(0, len(gates)):
                    N_samples = len(gates[m])
                    gate_sum = 0
                    for i in range(1, N_samples + 1):
                        gate_sum += IP_FW.iloc[i + gates[m][0] - 1, 0]
                    gate_avg = gate_sum / N_samples
                    IP_values.append(gate_avg)
                    mid_index = gates[m][N_samples // 2]
                    IP_times.append(mid_index)
                    
                # Creating a dataframe containing the ip values, DPID and the coordinates.    
                IP_data = pd.DataFrame(IP_values).T
                IP_data.columns = [f'ip{i + 1}' for i in range(IP_data.shape[1])]
                IP_data['DPID'] = DPID_current
                IP_data = pd.merge(IP_data, DPID_data[['DPID', 'A(x)', 'B(x)', 'M(x)', 'N(x)']], on='DPID', how='left')
                IP_data[['A(x)', 'B(x)', 'M(x)', 'N(x)']] = IP_data[['A(x)', 'B(x)', 'M(x)', 'N(x)']].astype('int') // electrode_separation

                results.append(IP_data)
        
                
        # ================================================= #
        #               Exporting the IP-data               #
        # ================================================= #
        
        # Append results to the output file. 
        if results:
            with open(output_file_path, 'a') as f:
                for result in results:
                    result.to_csv(f, sep='\t', header=f.tell()==0, index=False)
                    
        # Log the completion of file processing
        with open(processed_log_path, 'a') as log:
            log.write(f"{file_path}\n")

        # Free up memory
        del input_data, results, data_despiked, time, sections, subset_indices, u_subsets_gated, u_subsets_gated_indices
        gc.collect()

        logging.info(f"Finished processing file {file_path}")
        return results
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
            pool.map(process_file, batch_paths)
            
 # ================================================= #
 #              Conducting the despiking             #
 # ================================================= #           

if __name__ == '__main__':
    batch_size = batch_size  # Number of files to process in each batch
    logging.info(f"Starting processing with {len(file_list)} files and batch size of {batch_size}")

    # Read the log of processed files
    if os.path.exists(processed_log_path):
        with open(processed_log_path, 'r') as log:
            processed_files = log.read().splitlines()
    else:
        processed_files = []

    # Filter out the already processed files
    files_to_process = [file for file in file_list if file not in processed_files]

    logging.info(f"{len(files_to_process)} files left to process out of {len(file_list)}")

    # Initialize the output file (create or clear if exists)
    with open(output_file_path, 'w') as f:
        pass

    process_files_in_batches(files_to_process, batch_size)
    
    logging.info("Processing complete")
    logging.info(f"Results saved to {output_file_path}")