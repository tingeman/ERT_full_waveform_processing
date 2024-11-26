# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:02:13 2024

@author: Isabella Askjær Gaarde Lorenzen (s194835)
"""

# ================================================= #
#                Importing packages                 #
# ================================================= #

import numpy as np
import pandas as pd
import pygimli as pg
import os

# ================================================= #
#                   Introduction                    #
# ================================================= #

'''
This script exports the text file result of reprocessed IP-data into a DAT file
that can be imported into Aarhus Workbench. 

This script is written with assistance of ChatGPT.

This script connsiders datasets with different number of gates. If some data points 
is meassured with longer measurement times, the data points will have more gates
than the rest of the dataset. This code removes the excess ip gates if they are pressent
to ensure a consistancy in the exported data. 

In some cases the DAT file function at the end of this script is not activated upon
running the code. If no message regaring exporting the datafile occurs in the kernel,
please run the export function again.  

'''


# ================================================= #
#                 Input parameters                  #
# ================================================= #

# Please enter the electrode sepeartion from the survey. 
electrode_seperation = 5

# Please enter the first and the last ip window to export. 
start_ip_window = 1
end_ip_window = 13

# Please enter the delay and intergration time in seconds. The delay time is the first element in the list, and the
# intergration time, the subsequent values. 
delay_integration_time = [0.01, 0.02, 0.04, 0.04, 0.06, 0.06, 0.08, 0.1, 0.14, 0.18, 0.22, 0.28, 0.36, 0.404]

# Please enter the current on and current off time (e.g. 2, 2 s)
current_ON_OFF = [2, 2]

# Please enter the desired name of the .dat file, which will be exported after filtering.
filename = 'output_file2'

# Please enter the appropriate folder location for the exported filtered .dat file. 
folder_location = r'/Users/Isabe/Desktop/Ilulialik data processering/'

# This defines the final file location.  
file_location =  f"{folder_location}/{filename}.dat"

# Importing the .dat file with only resistivity data, to attach the new IP-data to. This can be retrieved
# using the Terrameter Toolbox software. 
DAT_file_ResOnly = pg.physics.ert.load("C:/Users/Isabe/Desktop/Regate IP-data/Ilulialik/Ilulialik_ResOnly.dat")

# To import the following text files corretly the following steps should be conducted in the textfile (e.g. in Notepad++):
    # 1) Replace all spacing with only one type of spacing: " ". 
    # 2) Make sure the header has the correct spacing of " ". 
    # 3) Remove end spaces (in Notepad++: 'Edit' -> 'Blank Operations' -> 'Trim Trailing Space')

# Importing the resulting text file from the script 1.2 Trend removal, stacking and gating. 
file_path = '/Users/Isabe/Desktop/Ilulialik data processering/gated_IP_data.txt'
dataset = pd.read_csv(file_path, delimiter="\t", header=None)

# Importing a text file containing Measurement ID, Data point ID (DPID), apparent resistivity (Ohm-m).
# This file prepares a text document needed to run the 2.1 Filter for IP-decay curve script. This script was written
# using a text file containing Measurement ID, Data point ID (DPID), apparent resistivity (Ohm-m) and IP-data from the instrument
# But in this script the ip-data from the instrument is naturally removed and the reprocessed IP-data attached. 
# The text file with the necessary information can be exported using Terrameter LS Toolbox. 
file_data = pd.read_csv('/Users/Isabe/Desktop/Ilulialik data processering/Filtering/Data for henfalskurver.txt',delimiter="\t", header=None)

# Importing a text file containing electrode coordinates with their respective data point ID. These data can be retrieve using
# the Terrameter LS Toolbox software. (MeasID, DPID, A(x), B(x), M(x) and N(x)). 
file_coordinates = pd.read_csv('/Users/Isabe/Desktop/Ilulialik data processering/Filtering/DPID og x koordinater for elektroder.txt',delimiter="\t", header=None)


# ================================================= #
#               Preparing processing                #
# ================================================= #

# Display settings of row/column in the kernel (preference for viewing data in the kernel)
pd.set_option("display.max_rows", 15)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


# Seperating the text files into individual columns. Dataset is imported as one string column containing all the data.
data = dataset[dataset.columns[0]].str.split(' ', expand=True) 
file_data = file_data[file_data.columns[0]].str.split(' ', expand=True)
file_coordinates = file_coordinates[file_coordinates.columns[0]].str.split(' ', expand=True)

# Assigning each column a name for the IP reprocessing result text file.
column_names = data.iloc[0]
data.columns = column_names
data = data[1:]

# Assigning each column a name for the file file_data. 
column_names = file_data.iloc[0]
file_data.columns = column_names
file_data = file_data[1:]

# Assigning each column a name for the file file_coordinates. 
column_names = file_coordinates.iloc[0]
file_coordinates.columns = column_names
file_coordinates = file_coordinates[1:]

# Remove columns that start with 'IP', to insure that any ip data that might be attached to the text file
# is removed. 
file_data = file_data.loc[:, ~file_data.columns.str.startswith('IP#')]


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
#               Preparing for export                #
# ================================================= #

# Creating a dataframe with the electrode coordinates from the input DAT file
columns_to_combine = ['a', 'b', 'm', 'n']
DAT_file_ResOnly_abmn = pd.concat([pd.DataFrame(DAT_file_ResOnly[col], columns=[col]) for col in columns_to_combine], axis=1)

# Converting the coordinate columns to numeric types
columns_to_convert = ['A(x)', 'B(x)', 'M(x)', 'N(x)']
data[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric, errors='coerce')


# Creating a list and then a dataframe where ip values are assigned to their respective DPID and coordinates
merge_list = []        
for index_data, row_data in data.iterrows():
    for index_DAT_file_ResOnly_abmn, row_DAT_file_ResOnly_abmn in DAT_file_ResOnly_abmn.iterrows():
        if (row_data['A(x)'] == row_DAT_file_ResOnly_abmn['a'] and 
            row_data['B(x)'] == row_DAT_file_ResOnly_abmn['b'] and 
            row_data['M(x)'] == row_DAT_file_ResOnly_abmn['m'] and 
            row_data['N(x)'] == row_DAT_file_ResOnly_abmn['n']):
            row = [row_DAT_file_ResOnly_abmn['a'], row_DAT_file_ResOnly_abmn['b'], row_DAT_file_ResOnly_abmn['m'], row_DAT_file_ResOnly_abmn['n'], row_data['DPID']]
            row.extend([row_data[ip] for ip in ip_columns])
            merge_list.append(row)
            print(row)

columns = ['a', 'b', 'm', 'n', 'DPID']
columns.extend(ip_columns)
df_merge_list = pd.DataFrame(merge_list, columns=columns)

# Rearranging the rows of the df_merge_list dataframe so they match the sorting of the rows in the input DAT file. 
DAT_file_ResOnly_abmn = pd.merge(DAT_file_ResOnly_abmn, df_merge_list, on=['a', 'b', 'm', 'n'], how='left', suffixes=('_orig', '_merge'))

# Replace empty strings of IP-data with NaN in ip_columns
DAT_file_ResOnly_abmn[ip_columns] = DAT_file_ResOnly_abmn[ip_columns].replace('', np.nan)

# Assign values to the 'ip' columns in input DAT file DAT_file_ResOnly.
for ip_col in ip_columns:
    DAT_file_ResOnly[ip_col] = [float(x) for x in DAT_file_ResOnly_abmn[ip_col] if x != '']

# Assign DPID to the input DAT file.     
DAT_file_ResOnly['DPID'] = [float(x) for x in DAT_file_ResOnly_abmn['DPID']]


# ================================================= #
#                Exporting text files               #
# ================================================= #

# Exporing the text file containing MeasID, DPID, resitivity and ip-windows. 

# Retrieve the ip columns with their DPID, after filtering some rows in the step above to match the 
# data points that will be exported in the DAT file. 
ip_columns = [col for col in DAT_file_ResOnly_abmn.columns if col.startswith('ip')]
filtered_ip_data = DAT_file_ResOnly_abmn[['DPID'] + ip_columns]

# Merge `filtered_ip_data` with `file_data` based on 'DPID', to append the reprocessed IP-data. 
data_to_export1 = file_data.merge(filtered_ip_data, on='DPID', how='left')

# Rename the IP columns to "IP#1(mV/V)", "IP#2(mV/V)", etc.
for i, col in enumerate(ip_columns, start=1):
    data_to_export1.rename(columns={col: f'IP#{i}(mV/V)'}, inplace=True)

# Further remove any data points where the IP does not have value.     
data_to_export1 = data_to_export1.dropna(subset=['IP#1(mV/V)'])

# Exporting the text file 
filename_text1 = 'Data_for_henfaldskurver_trimmet'
file_location1 =  f"{folder_location}/{filename_text1}.txt"
data_to_export1.to_csv(file_location1, index=False)


# Exporting the text file containing MeasID, DPID and coordinates. 

# Extracting DPID values from DAT_file_ResOnly_abmn
valid_dpid = DAT_file_ResOnly_abmn['DPID']

# Filter file_coordinates to keep only rows with DPID values present in valid_dpid
file_coordinates = file_coordinates[file_coordinates['DPID'].isin(valid_dpid)]

# DPID with NaN values in IP columns is defined
dpid_with_nan_ip1 = DAT_file_ResOnly_abmn[DAT_file_ResOnly_abmn['ip1'].isna()]['DPID']

# Rows where the DPID is Nan is removed. 
file_coordinates = file_coordinates[~file_coordinates['DPID'].isin(dpid_with_nan_ip1)]
data_to_export2 = file_coordinates.dropna(subset=['DPID'])

# Exporting the text file 
filename_text2 = 'DPID_og_koordinater_for_elektroder_trimmet'
file_location2 =  f"{folder_location}/{filename_text2}.txt"
data_to_export2.to_csv(file_location2, index=False)


# ================================================= #
#        Exporting the data into a DAT file         #
# ================================================= #

def exportRes2dInv(data, start_ip_window, end_ip_window, delay_integration_time, filename="out.res2dinv", ar_idfy=11, sep='\t',
                   arrayName='mixed array', rhoa=False, verbose=False):
    """Save data file under res2dinv general array format."""
    x = [np.round(ii[0], decimals=2) for ii in data.sensorPositions()]
    y = [np.round(ii[1], decimals=2) for ii in data.sensorPositions()]
    if not np.any(y):
        y = [np.round(ii[2], decimals=2) for ii in data.sensorPositions()]

    dist = [np.sqrt((x[ii]-x[ii-1])**2 + (y[ii]-y[ii-1])**2)
            for ii in np.arange(1, len(y))]
    dist2 = [x[ii]-x[ii-1] for ii in np.arange(1, len(y))]
    print(min(dist), min(dist2))
    # %% check for resistance or resistivity
    if (data.allNonZero('r') or data.allNonZero('R')) and not rhoa:
        datType = '1'
        res = data('r')
    elif data.allNonZero('rhoa'):
        datType = '0'
        res = data('rhoa')
    else:
        raise BaseException("No valid apparent resistivity data!")
    # %% check for ip
    ip_exists = all(data.haveData(f'ip{j}') for j in range(start_ip_window, end_ip_window + 1))
    
    # %% write res2Dinv file
    with open(filename, 'w') as fi:
        fi.write(arrayName+'\n')
        fi.write(str(np.round(min(dist2), decimals=2))+'\n')
        fi.write(str(ar_idfy)+'\n')
        fi.write('15\n')
        fi.write('Type of measurement (0=app.resistivity,1=resistance) \n')
        fi.write(datType+'\n')
        fi.write(str(DAT_file_ResOnly_abmn['ip1'].dropna().count())+'\n') # Retrieving the number of data points in the output file
        fi.write('2'+'\n')
        if ip_exists:
            fi.write('11\n')
            fi.write('Chargeability\n')
            fi.write('mV/V\n')
            delay_integration_time = ' '.join(map(str, delay_integration_time))
            fi.write(delay_integration_time)  # delay/integration time
            fi.write("\n")
            lines = []
            for oo in range(len(data('a'))):
                ip_values = [data(f'ip{j}')[oo] for j in range(start_ip_window, end_ip_window + 1)]
                if any(np.isnan(ip_values)):
                    continue  # Skip rows where any IP value is NaN
                ip_values_str = [str(ip) for ip in ip_values]
                line = '4' + sep + \
                       str(x[int(data('a')[oo])]) + sep + \
                       str(y[int(data('a')[oo])]) + sep + \
                       str(x[int(data('b')[oo])]) + sep + \
                       str(y[int(data('b')[oo])]) + sep + \
                       str(x[int(data('m')[oo])]) + sep + \
                       str(y[int(data('m')[oo])]) + sep + \
                       str(x[int(data('n')[oo])]) + sep + \
                       str(y[int(data('n')[oo])]) + sep + \
                       str(res[oo]) + sep + \
                       sep.join(ip_values_str)
                lines.append(line)
        else:
            fi.write('0\n')
            lines = ['4' + sep +
                     str(x[int(data('a')[oo])]) + sep +
                     str(y[int(data('a')[oo])]) + sep +
                     str(x[int(data('b')[oo])]) + sep +
                     str(y[int(data('b')[oo])]) + sep +
                     str(x[int(data('m')[oo])]) + sep +
                     str(y[int(data('m')[oo])]) + sep +
                     str(x[int(data('n')[oo])]) + sep +
                     str(y[int(data('n')[oo])]) + sep + 
                     str(res[oo])
                     for oo in range(len(data('a')))]

        fi.writelines("%s\n" % l for l in lines)

# This function will export the filtered .dat file (after the second filtering step), if it does not already exist in the chosen folder. 
def export_if_not_exists(filtered_data, start_ip_window, end_ip_window, delay_integration_time, file_location):
    if not os.path.exists(file_location):
        exportRes2dInv(filtered_data, start_ip_window, end_ip_window, delay_integration_time, file_location)
        print("The filtered data file was successfully exported.")
    else:
        print("File already exists at the destination. Skipping export.")

# Executing the export if the file does not already exist. 
export_if_not_exists(DAT_file_ResOnly, start_ip_window, end_ip_window, [len(ip_columns)] + delay_integration_time + current_ON_OFF, file_location)

