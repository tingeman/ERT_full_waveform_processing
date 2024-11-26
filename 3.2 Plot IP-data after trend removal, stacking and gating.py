# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:02:13 2024

@author: Isabella Askjær Gaarde Lorenzen (s194835)
"""

# ================================================= #
#                Importing packages                 #
# ================================================= #

import pandas as pd
from matplotlib import pyplot as plt

# ================================================= #
#                   Introduction                    #
# ================================================= #

'''
This script subsequently plots the reprocessed gated IP-decay curve from the results 
of the script 1.3 Exported resprocessed IP-data into a DAT file.  

This script is written with assistance from ChatGPT

'''


# ================================================= #
#                 Input parameters                  #
# ================================================= #

# Please enter the time values of which the value of each IP window is plotted in ms. 
# This time value is the middle value of each intergrated time window. 
IP_times = [10, 40, 80, 130, 190, 260, 350, 470, 630, 830, 1080, 1400, 1782]

# Importing the text file from the reprocessing of the IP-data from the script 1.2 Trend removal, stacking and gating. 
file_path = '/Users/Isabe/Desktop/Ilulialik data processering/gated_IP_data.txt'
dataset = pd.read_csv(file_path, delimiter="\t", header=None)


# ================================================= #
#                   Preparing data                  #
# ================================================= #

# Display settings of row/column in the kernel (preference for viewing data in the kernel)
pd.set_option("display.max_rows", 15)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# Separating the text files into individual columns. Dataset is imported as one string column containing all the data.
data = dataset[dataset.columns[0]].str.split(' ', expand=True)

# Assigning each column a name for the first text file.
column_names = data.iloc[0]
data.columns = column_names
data = data[1:]

# =========================================================================== #
#  Preparing a dataframe containing appropriate IP-data, DPID and coordinates #
# =========================================================================== #

# This function retrieves the last five columns from the resulting file from the reprocessing of the IP-data
# This step is applied to accomodate potentially different number of gates in the dataset. 
def last_n_non_none_values(row, n):
    non_none_values = row.dropna().values
    return non_none_values[-n:]

# Apply the function to each row
result = data.apply(lambda row: last_n_non_none_values(row, 5), axis=1)

# Retrieve the appropriate column names for the result above. 
DPID_coordinates_col_name = [col for col in data.columns if col is not None][-5:]

# Create a new DataFrame with the results.
DPID_coordinates = pd.DataFrame(result.tolist(), index=data.index, columns=DPID_coordinates_col_name)

# Creates a list of the data point id's in the DPID_coordinates df. 
DPID_list = DPID_coordinates['DPID'].tolist()

# Compute a list of ip columns to be exported. 
ip_columns = [col for col in data.columns if col and col.startswith('ip')]

# Retrieving the columns with ip-data and negleting the ip-data that exceeds the
# list if the ip columns. That means if the initial rows have 13 ip windows but some
# arbitrary rows have 16 ip windows, the last 3 ip windows will be neglected. 
ip_data = data.iloc[:, :len(ip_columns)]


# Copying the ip data
IP_data = ip_data.copy()

# Identifies any rows where the ip-data is incorrectly assigned as a DPID in the DPID_coordinates
# due to the reprocessed IP-data exporting mistakingly too few windows. 
rows_to_replace = IP_data.isin(DPID_list).any(axis=1)

# For the identified data points with too few IP-windows, the ip values are replaced with empty space,
# to insure it is not included in the exported file. 
IP_data.loc[rows_to_replace, :] = ""

# The IP-data from the reprocessing and the DPID and the coordinates are finally merged into one df. 
data = pd.merge(IP_data, DPID_coordinates, left_index=True, right_index=True)



# ================================================= #
#              Plotting of gated IP-data            #
# ================================================= #

# Creates a list of the DPID from the resulting IP-data file after the trend removal, stacking and gating,
# If the IP values are empty (can happen to a few data points), the plotting of that data point is skipped. 
dpid_list = []

# Loop through each column in the DataFrame
for column in data.columns:
    # Check if the column name starts with 'ip'
    if column.startswith('ip'):
        # Check for empty string values in the column
        empty_dpid = data[data[column] == '']['DPID']
        # Append the DPID values to the list
        dpid_list.extend(empty_dpid)
        
# Remove duplicates from the list
dpid_list = list(set(dpid_list))



# Plots the IP-curves individually
for i in range(0, len(data)):    
    # Retrieve the current DPID
    dpid = data['DPID'].iloc[i]
    
    print(f'Plotting the IP-decay curve for DPID {dpid}')
    
    # Skip rows with DPID in dpid_list
    if dpid in dpid_list:
        continue
    
    # Retrieve the current IP-values in the df and convert from string to float. 
    Y_value = data[ip_columns].iloc[i].tolist()
    Y_value = [float(value) for value in Y_value]
    
    # Plot the gate values
    plt.plot(IP_times, Y_value, marker='o', linestyle='-', label='Gate Values')
    plt.xlabel('Time [ms]')
    plt.ylabel('IP values [mV/V]')
    plt.title(f'IP decay curve for DPID {dpid}')
    plt.legend()
    plt.grid(True)
    plt.show()
    

