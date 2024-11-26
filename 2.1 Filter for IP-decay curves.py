# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:14:42 2024

@author: Isabella Askjær Gaarde Lorenzen (s194835)
"""
# ================================================= #
#                Importing packages                 #
# ================================================= #

import pygimli as pg
import pybert as pb
from pybert.importer.exportData import exportRes2dInv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
from scipy.optimize import minimize
import statistics
import os

# ================================================= #
#                   Introduction                    #
# ================================================= #

"""
This script is written with assistance from ChatGPT, and is based on the article:
Adrian Flores Orozco et al. ‘Decay curve analysis for data error quantification in time-domain
induced polarization imaging’. In: Geophysics 83 (2 Mar. 2018), E75–E86. issn: 19422156.
doi: 10.1190/geo2016-0714.1     

This script is written using survey data collected with the multi gradient protocol (MG). 
If data collected with another protocol is to be filtered using this script, adjustments 
should be made accordingly, or errors might occur. 


This script assumes the same IP-delay and integration times for IP-windows for the whole dataset. 
If IP-delay and integration times for IP-windows varies over the dataset adjustments to the script 
should be made accordingly, or alternativly filter data points with each setting seperatly. 

This script automatically neglects the first IP-window, due to general erratic behavior in the 
first IP-window in the test-dataset. Erratic behavior can be due to a spike in the electromagnetic
field, that happens when the current is suddently switched off or reversed 
[1]. Usually the IP-delay time is applied to accomodate
this issue but if the IP-delay time is too short for the local conditions, EMF spikes can be
dominant in the data. If the first IP-window in the IP-decay curves generally looks acceptable 
(this can be assessed using the Terrameter LS Toolbox), the first IP-window should be included 
in this filter. In that case please follow the instructions in the section "Removing IP-windows"
five sections below. 

Disclaimer: This filter has only been tested using one dataset, why results of filtering other dataset
should be examined. It is recommended by this author to take a look at the chargeabilty and rmsd plots,
before and after the two filtering steps, and assess the quality of the filtering. It is here important to 
note that the rmsd (root mean square deviation) plot is not necessarily an indicator of good or poor data
quality but only an indication of how well the power law model is fitted to the measured IP-decay curves. 
It is also important to note that visual expection of the curve fit proved that the root mean square deviation, 
can also be small for IP-decay curves with very erratic behavior, which are subjects to random erros. It is
as well important to note that the rmsd plot, does not necessarily indicate systematic errors, of good looking 
ip-decay curves, which are magnatudes larger or smaller than the general decay tendancy. 
This author also urges the user to examine the histogram filter. This filter only seperates the data points
by empty bins from the main distribution of valid data points. If the main distibution
takes the shape of a normal distribution but is abrupted by bins of very small sizes, the bins of small sizes
could also work as a seperation of the main destibution. In that case the script could be adjusted to accomondate
the specific case. 

In some cases, when the script is run the first time, the visualization plots of the raw chargeability data and the
correspinding rmsd plot, is plotted incorrectly. If the first chargeability/rmsd plot looks significantly different
than the second and third plot, please run the plotting and visualization section again. This will fix the issue. 

Before running this code, please enter the appropriate elements in the sections "Defining export file name and location", 
" Adjustable elements specific to the respective survey", "Importing data files" and "Removing IP-windows". 

Good luck!

"""

# ================================================= #
#       Defining export file name and location      #
# ================================================= #
# In this section, please choose an appropriate name and location for the file, which will be exported after the filtering process. 

# Please enter the desired name of the .dat file, which will be exported after filtering.
filename = 'Ilulialik_filtered_TEST'

# Please enter the appropriate folder location for the exported filtered .dat file. 
folder_location = '/Users/Isabe/Desktop/Ilulialik data processering/Filtering'

# This defines the final file location.  
file_location =  f"{folder_location}/{filename}.dat"


# ======================================================= #
#  Adjustable elements specific to the respective survey  #
# ======================================================= #

# Please enter the electrode seperation in meteres.
electrode_seperation = 5

# Please enter in meters the beginning and end of the survey profile (used for plotting the pseudosection). 
profile_start = 0
profile_end  = 900


# Please enter the delay time (first element) and the integration times in seconds (can be retrieved in the script 
# '4.1 Full reprocessing of a signle data point (including plots)'). 
delay_integration_time = [0.01, 0.02, 0.04, 0.04, 0.06, 0.06, 0.08, 0.1, 0.14, 0.18, 0.22, 0.28, 0.36, 0.403]

# Please enter the time at the middle of each gate in ms (used for plptting the decay curves).
Integration_times = [10, 40, 80, 130, 190, 260, 350, 470, 630, 830, 1080, 1400, 1782]


# Please enter the current on and current off time (e.g. 2, 2 s)
current_ON_OFF = [2, 2]


# ================================================= #
#               Importing data files                #
# ================================================= #
# In this section, please enter the appropriate file locations. 

# To import the following text files corretly the following steps should be conducted in the textfile (e.g. in Notepad++):
    # 1) Replace all spacing with only one type of spacing: " ". 
    # 2) Make sure the header has the correct spacing of " ". 
    # 3) Remove end spaces (in Notepad++: 'Edit' -> 'Blank Operations' -> 'Trim Trailing Space')
    
# The following two text files has been exported using Terrameter LS Toolbox. It is important to only import text files
# containing the data listed for each text file. 

# Importing a text file containing Measurement ID, Data point ID (DPID), apparent resistivity (Ohm-m), and each IP-window. 
dataset = pd.read_csv('/Users/Isabe/Desktop/Ilulialik data processering/Filtering/Data_for_henfaldskurver_trimmet.txt',delimiter="\t", header=None)

# Importing a text file containing electrode coordinates with their respective data point ID 
# (MeasID, DPID, A(x), B(x), M(x) and N(x)) 
dataset_coordinates = pd.read_csv('/Users/Isabe/Desktop/Ilulialik data processering/Filtering/DPID_og_koordinater_for_elektroder_trimmet.txt',delimiter="\t", header=None)

# Importing the .dat file for applying the filters and visualizing data, and finally exporting the filtered data. 
dataOriginal = pg.physics.ert.load("/Users/Isabe/Desktop/Ilulialik data processering/output_file.dat")


# ================================================= #
#            Initial preparing of data              #
# ================================================= #
# Display settings of row/column in the kernel (preference for viewing data in the kernel)
pd.set_option("display.max_rows", 15)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# Seperating the text files into individual columns. Dataset is imported as one string column containing all the data. 
data = dataset[dataset.columns[0]].str.split(' ', expand=True)
data_coordinates = dataset_coordinates[dataset_coordinates.columns[0]].str.split(' ', expand=True)

# Assigning each column a name for the first text file. 
column_names = data.iloc[0]
data.columns = column_names
data = data[1:]

# Assigning each column a name for the second text file. 
column_names = data_coordinates.iloc[0]
data_coordinates.columns = column_names
data_coordinates = data_coordinates[1:]

# Converting IP-columns in the first text file into float types.
ip_columns = [col for col in data.columns if col.startswith('IP#')]
data[ip_columns] = data[ip_columns].astype(float)


# ================================================= #
#                 Removing IP-windows               #
# ================================================= #

# This section neglects the first IP-window. To remove more than one IP window, 
# few adjustments in this section should be made. To include the first IP-window
# please follow the instructions on line 140 further below in this section. 

# Defining the range of IP-windows to include.
# IMPORTANT: Make sure to enter the correct end_ip_window value. This should be the 
# number of the last ip-window incluted in the filtering. 
start_ip_window = 1
end_ip_window = 13

# Removing the first IP-window from the dataset. 
#Remove_first_IP_Window = [item for item in data if item != 'IP#1(mV/V)']
#data = data[Remove_first_IP_Window]

# Removing the corresponding first integration time.
#X_IP_Window = Integration_times[1:]

# If the first ip-window is to be incluted in the filtering process, please change the
# value of start_ip_window to 1 and enter a '#' in front of line 134, 135 and 138 above, 
# and remove the '#' in front of line 144, below. 

X_IP_Window = Integration_times


# ===================================================================== #
#  Power law fit of IP-decay curves and the associated goodness of fit  #
# ===================================================================== #

# To include negative IP-data which are mirrored along the x-axis of the IP-decay curve plots, a function is defined for both +a and -a.
# Power law model in the first quadrant.
def power_law(x, a, b, c):
    return a * safe_exp(-b * np.log(x)) + c
    # Where a = α, b = β and c = ε

# Power law model in the fourth quadrant.
def negative_power_law(x, a, b, c):
    return -a * safe_exp(-b * np.log(x)) + c
    # Where a = α, b = β and c = ε

# Safe exponential function to prevent overflow.
def safe_exp(x):
    return np.exp(np.clip(x, -700, 700))
 
# Initializing empty lists to store data from the for-loop below. 
fitted_params_list = []
RMSD_list = []
Measured_IP_values_list = []
Fitted_IP_values_list = []
large_a_values_list = []

# Creating a list of each DPIP. 
unique_dpid_values = data["DPID"].unique()

# This loop iterates over each data point and fits a positive or negative power law model to the IP-decay curve, and plots the measured and fitted IP-decay curve. 
# This loop also computes the parameters of the fitted curve, the goodness of fit between the curves (rmsd), and a list of the fitted IP-values for each IP-window. 
for dpid in unique_dpid_values:
    # Selecting rows corresponding to the current DPID
    DPID = data["DPID"] == dpid
    DCData = data[DPID] # Short for decay curve data
    index = DCData.index.tolist()
    
    # Selecting only the values from columns starting with "IP#"
    ip_columns = [col for col in DCData.columns if col.startswith('IP#')]
    ip_values = DCData[ip_columns].astype(float) 
    Y_IP_values = ip_values.loc[index[0]].tolist()    
    
    # Defining x for displaying a smooth fitted curve in the plot. 
    x_fit = np.linspace(min(X_IP_Window), max(X_IP_Window), 100)
    
    # Performing curve fitting for positive IP-values (+a, decay in 1st quadrant) and for negative IP-values (-a, decay in 4th quadrant).
    try:
        # Curve fitting with the power law model is conducted in this block. 
        # The block ignores the covariance warning, which can be due to poor fit, noisy data or too large or small parameters.
        # Since the fits are conducted on unfiltered data with expected outliers, the covarience warning can be expected, and will not influence the fittings itself. 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OptimizeWarning)
            
            # Curve fitting with both the positive and negative power law model is conducted for the current data point. 
            popt_pos, pcov_pos = curve_fit(power_law, X_IP_Window, Y_IP_values, maxfev=100000)
            popt_neg, pcov_neg = curve_fit(negative_power_law, X_IP_Window, Y_IP_values, maxfev=100000)

        # Computing the fitted IP-values for both the positive and negative fitted power law models. 
        y_fit_pos = power_law(X_IP_Window, *popt_pos)
        y_fit_neg = negative_power_law(X_IP_Window, *popt_neg)
        
        # Calculating the root mean square deviation (RMSD) for both the positive and negative power law fits.
        rmsd_pos = np.sqrt(np.mean((Y_IP_values - y_fit_pos) ** 2))
        rmsd_neg = np.sqrt(np.mean((Y_IP_values - y_fit_neg) ** 2))

        # Choosing the positive or negative model for the current data point as the model with the smallest RMSD. 
        if rmsd_pos < rmsd_neg:
            popt = popt_pos
            y_fit_values = y_fit_pos                        # Used for a list of fitted IP-values.
            y_fit = power_law(x_fit, *popt_pos)             # Used for a smooth display of fitted curves.
            rmsd = rmsd_pos
            PosOrNegCurve = 'positive'
            print(f"A fitted curve in the first quadrant was selected, where RMSD_Q1 = {rmsd_pos:.2f} < RMSD_Q4 = {rmsd_neg:.2f}")
        else:
            popt = popt_neg
            y_fit_values = y_fit_neg                        # Used for list of fitted IP-values
            y_fit = negative_power_law(x_fit, *popt_neg)    # Used for a smooth looking fitted curve
            rmsd = rmsd_neg
            PosOrNegCurve = 'negative'
            print(f"A fitted curve in the fourth quadrant was selected, where RMSD_Q1 = {rmsd_pos:.2f} > RMSD_Q4 = {rmsd_neg:.2f}")
            
    except RuntimeWarning:
        warnings.warn("Curve fitting encountered a RuntimeWarning")
    
    
    # Defining the current measure ID and data point ID for lists below. 
    DPID_value = DCData["DPID"].tolist()[0]
    MeasID_value = DCData["MeasID"].tolist()[0]
        
    # Storing fitting parameters in a list.
    param_list = [MeasID_value, DPID_value, popt[0], popt[1], popt[2]]
    fitted_params_list.append(param_list)
    
    # Storing RMSD values in a list.
    rmsd_ID = [MeasID_value, DPID_value, rmsd, PosOrNegCurve]
    RMSD_list.append(rmsd_ID)
    
    # Storing measured IP values in a list.
    MIP_list = [MeasID_value, DPID_value, *Y_IP_values]
    Measured_IP_values_list.append(MIP_list)
    
    # Storing fitted IP values in a list. 
    FIP_list = [MeasID_value, DPID_value, *y_fit_values, PosOrNegCurve]
    Fitted_IP_values_list.append(FIP_list)
    
    
    # Plotting the measured IP-curve. 
    plt.plot(X_IP_Window, Y_IP_values, color='dodgerblue', label='Measured decay curve')

    # Plotting the fitted curve.
    plt.plot(x_fit, y_fit, '--', color='red', label='$m_{f}$(t)=α$t^{-β}$+ε')


# ================================================================== #
#  Creating DataFrames of the sorted data from the previous section  #
# ================================================================== #

# Assigning fitting parameters to a dataframe with measurement ID and data point ID. 
df_fitted_params = pd.DataFrame(fitted_params_list, columns=['MeasID', 'DPID','α', 'β', 'ε'])

# Assigning RMSD values to a dataframe with measurement ID, data point ID and curve type. 
df_rmsd = pd.DataFrame(RMSD_list, columns=['MeasID', 'DPID', 'RMSD', 'curve type'])

# Assigning measured IP values to a dataframe with measurement ID and data point ID.
df_MIP = pd.DataFrame(Measured_IP_values_list, columns=[['MeasID', 'DPID'] + ip_columns])

# Assigning fitted IP values to a dataframe with measurement ID, data point ID and curve type. 
df_FIP = pd.DataFrame(Fitted_IP_values_list, columns=[['MeasID', 'DPID'] + ip_columns + ['curve type']])


# =================================================== #
#  Calculating the total chargeability M for plotting #
# =================================================== #

# The total chargeability is calculated as a temporal average of the voltage decay [2], to plot a representative 
# IP-pseudosection during the filtering steps. Note that the calculation of the total chargeability averages out erratic 
# behavior of the decay curve [2], but this calculation will not effect the filtering process, and is solely used to 
# plot the IP-psudosection to view obvious outliers and filtered data points. 

# Creating a list of IP-values using the .dat file.      
mi_IP_list = [[dataOriginal[f'ip{j}'][i] for j in range(start_ip_window, end_ip_window + 1)] for i in range(len(dataOriginal['ip']))]


# Calculating the total chargeability M as the temporal average of the decay curve.
M_list = []
for i in range(len(dataOriginal['ip'])):
    # Calculating the weighted sum of products (temporal average)
    temporal_sum = sum(mi * time for mi, time in zip(mi_IP_list[i], X_IP_Window))
    
    # Calculating the sum of IP-window duration (time).
    sum_intervals = sum(X_IP_Window)
    
    # Calculating the temporal average for the total chargeability M.
    temporal_average = temporal_sum / sum_intervals
    M_list.append(temporal_average)

# Assigning the total chargeability M to the .dat file to prepare for plotting.
dataOriginal['TCM'] = M_list # TCM = Total Chargeability M

# Defining the minimum and maximum total chargeability for plotting purpurses. 
TCM_min = min(M_list)
TCM_max = max(M_list)


# =================================================== #
#  Preparing the goodness of fit (rmsd) for plotting  #
# =================================================== #

# Converting rmsd to the logarithm, log10(rmsd).
df_rmsd['RMSD'] = np.log10(df_rmsd['RMSD'])

# Defining the minimum and maximum rmsd for plotting purpurses. 
rmsd_min = min(df_rmsd['RMSD'])
rmsd_max = max(df_rmsd['RMSD'])

# Creating a new dataframe where rmsd along with curve type is assigned to its respective electrode coordinate based on the DPID. 
df_rmsd_coor = pd.merge(data_coordinates, df_rmsd, on='DPID')

# Convert columns to numeric
columns_to_convert = ['A(x)', 'B(x)', 'M(x)', 'N(x)']
for col in columns_to_convert:
    df_rmsd_coor[col] = pd.to_numeric(df_rmsd_coor[col], errors='coerce')


# Check electrode_seperation and convert values to floats if it equals 0.5
if electrode_seperation == 0.5:
    for col in ['A(x)', 'B(x)', 'M(x)', 'N(x)']:
        df_rmsd_coor[col] = df_rmsd_coor[col].astype(float)


# Creating a dataframe with the electrode coordinates from the dataOriginal file.
columns_to_combine = ['a', 'b', 'm', 'n']
dataOriginal_abmn = pd.concat([pd.DataFrame(dataOriginal[col], columns=[col]) for col in columns_to_combine], axis=1)


# Multiply the columns by the electrode separation
dataOriginal_abmn[['a', 'b', 'm', 'n']] = dataOriginal_abmn[['a', 'b', 'm', 'n']] * electrode_seperation


# Creating a list and then a dataframe where rmsd, DPID and curve type is assigned to their respective coordinates.
merge_list = []        
for index_rmsd, row_rmsd in df_rmsd_coor.iterrows():
    for index_dataOriginal_abmn, row_dataOriginal_abmn in dataOriginal_abmn.iterrows():
        if (row_rmsd['A(x)'] == row_dataOriginal_abmn['a'] and 
            row_rmsd['B(x)'] == row_dataOriginal_abmn['b'] and 
            row_rmsd['M(x)'] == row_dataOriginal_abmn['m'] and 
            row_rmsd['N(x)'] == row_dataOriginal_abmn['n']):
            merge_list.append([row_dataOriginal_abmn['a'], row_dataOriginal_abmn['b'], row_dataOriginal_abmn['m'], row_dataOriginal_abmn['n'], row_rmsd['DPID'], row_rmsd['RMSD'], row_rmsd['curve type']])
            print([row_dataOriginal_abmn['a'], row_dataOriginal_abmn['b'], row_dataOriginal_abmn['m'], row_dataOriginal_abmn['n'], row_rmsd['DPID'], row_rmsd['RMSD'], row_rmsd['curve type']])           
df_merge_list = pd.DataFrame(merge_list, columns = ['a', 'b', 'm', 'n', 'DPID', 'RMSD', 'curve type'])

# Rearranging the rows of the df_merge_list dataframe to they match the sorting of the rows in the dataOriginal file.
dataOriginal_abmn = pd.merge(dataOriginal_abmn, df_merge_list[['a', 'b', 'm', 'n', 'DPID', 'RMSD', 'curve type']], on=['a', 'b', 'm', 'n'], how='left', suffixes=('_orig', '_merge'))

# Adding rmsd and DPID to the dataOriginal file. This will make it possible to plot rmsd in the pseudosection.
dataOriginal['DPID'] = [int(x) for x in dataOriginal_abmn['DPID'].tolist()]
dataOriginal['RMSD'] = [float(x) for x in dataOriginal_abmn['RMSD'].tolist()]


# ================================================================= #
#  Analysis of the spatial consistancy of the recorded decay curve  #
# ================================================================= #

# This section detects and filters outlier data points by computing a reference curve for each injection electrode combination,
# and moving the reference curve along the vertical axis, computes the vertical shift for the referance curve to each fitted 
# IP-decay curve within the electrode combination, and filters out the data points where the shift is too great. 

# Creating a list to store IP coordinates
IP_coordinates_list = []

# Iterating over each row in df_FIP (dataframe for IP-values of the fitteed curves).
for index_FIP, row_FIP in df_FIP.iterrows():
    # Iterating over each row in dataOriginal_abmn.
    for index_dataOriginal_abmn, row_dataOriginal_abmn in dataOriginal_abmn.iterrows():
        # Checking if DPID matches. 
        if row_FIP['DPID'] == row_dataOriginal_abmn['DPID']:
            # Preparing a list to store IP values for the current row. 
            ip_values = [row_dataOriginal_abmn['a'], row_dataOriginal_abmn['b'], row_dataOriginal_abmn['m'], row_dataOriginal_abmn['n'], row_dataOriginal_abmn['DPID'], row_FIP['MeasID'], row_dataOriginal_abmn['RMSD'], row_dataOriginal_abmn['curve type']]
            # Iterating over the range of IP windows. 
            for i in range(start_ip_window, end_ip_window + 1):
                ip_column_name = f'IP#{i}(mV/V)'
                # Checking if the IP window column exists.
                if ip_column_name in row_FIP:
                    ip_values.append(row_FIP[ip_column_name])                    
                else:
                    ip_values.append(None)
            # Appending the list of coordinates, DPID, rmsd, curve type and IP-values for the current row.          
            IP_coordinates_list.append(ip_values)
            print(ip_values)

# Defining the columns for the dataframe below. 
columns = ['a', 'b', 'm', 'n', 'DPID', 'MeasID', 'RMSD', 'curve type'] + [f'IP#{i}(mV/V)' for i in range(start_ip_window, end_ip_window + 1)]

# Creating a dataframe of the coordinates, DPID, rmsd, curve type and IP-values from the block above. 
df_IP_coordinates = pd.DataFrame(IP_coordinates_list, columns=columns)


# >> The following block computes: 
        # 1) The reference curve for the negative and/or positive curve type for each injection electrode (A(x) and B(x)) combination.
        # 2) The shift value, which is a quantification of the vertical shift of each reference curves to each fitted decay curves with 
        #    the same injection electrode combination, which results in the smallest rmsd. 

# Creating an empty list to store dataframes with updated shift values.
concatenated_dfs = []

# Creating an empty list of injection electrode combinations, which will be updated with unique combinations during the looping.  
processed_combinations = set()

# Iterating over each unique injection electrode combination. 
for index, row in df_IP_coordinates.iterrows():
    # Retreives the injection electrodes in the current row.
    current_combination = (row['a'], row['b'])
    # Runs the code for each unique injection electrode combination. 
    if current_combination not in processed_combinations:
        # >> Plotting all the fitted curves from the current electrode combination
        
        # Slices the rows in the df_IP_coordinates dataframe using the current injection electrode combination. 
        sliced_rows = df_IP_coordinates[(df_IP_coordinates['a'] == row['a']) & (df_IP_coordinates['b'] == row['b'])]

        # Selecting DPIDs of the sliced rows.
        unique_dpid_values_ref = sliced_rows["DPID"].unique()

        # Iterating over each data point using DPID, and computes a plot with all the fitted curves from the current electrode combination. 
        for i, dpid in enumerate(unique_dpid_values_ref):
            # Selecting the row corresponding to the current data point.
            DPID = sliced_rows["DPID"] == dpid
            current_row = sliced_rows[DPID]
            index = current_row.index.tolist()
            
            # Selecting only the values in the row from columns starting with "IP#"
            ip_columns_ref = [col for col in current_row.columns if col.startswith('IP#')]
            ip_values_ref = current_row[ip_columns_ref].astype(float)
            Y_IP_values_ref = ip_values_ref.loc[index[0]].tolist()

            # Plotting the fitted curve of the current data point.
            plt.plot(X_IP_Window, Y_IP_values_ref, '-', label=f'DPID: {dpid}')
            
            # Remove the '#' from the appropriate lines below to inspect the rmsd and curve fitting parameters in the plots of the injection electrode combinations. 
            # Defining the rmsd for the data point
            #rmsd_dpid = sliced_rows[DPID]["RMSD"].tolist()[0]

            #a = df_fitted_params[df_fitted_params["DPID"] == dpid]['α'].tolist()[0]
            #b = df_fitted_params[df_fitted_params["DPID"] == dpid]['β'].tolist()[0]
            #c = df_fitted_params[df_fitted_params["DPID"] == dpid]['ε'].tolist()[0]

            # Define position for text box dynamically
            #text_x = 0.99  # Adjust as needed
            #text_y = 0.81 - i * 0.117  # Adjust as needed

            #plt.text(text_x, text_y, f'DPID: {dpid}\nRMSD: {rmsd_dpid:.3}\nα: {a:.3}, β: {b:.3}, ε: {c:.3} ', fontsize=7, ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

        # >> Computing the referance curve for the current electrode injection combination.        
        
        # A function which computes the median values of each IP-window of M-fitted curves for both curve types.
        def referance_curves(sliced_rows, curve_type):
            curves = sliced_rows[sliced_rows["curve type"] == curve_type]
            if curves.empty:
                return None  
            
            # Retrieving the fitted IP-values for the current curve type. 
            ip_values_ref = curves[ip_columns_ref].astype(float)
            
            # Computing the median value of each IP-window for each of the curve types.
            # These values constitues the refrerence curve(s). 
            median_values = ip_values_ref.median()  
            all_median_values = []
            all_median_values.append(median_values)
            all_median_values_df = pd.DataFrame(all_median_values)
            overall_median_values = all_median_values_df.median().tolist()
            
            return overall_median_values

        # Defining the final IP values for the reference curves for the current electrode injection combination. 
        NegCurves_referance_curve = referance_curves(sliced_rows, 'negative')
        PosCurves_referance_curve = referance_curves(sliced_rows, 'positive')

        # Plotting only the referance curve of the positive or negative curve type if they are available.
        # E.g. if no data with the negative curve type is present for the current electrode injection combination, 
        # only the referance curve with the positive curve type will be plotted. 
        if NegCurves_referance_curve is not None:
            plt.plot(X_IP_Window, NegCurves_referance_curve, '--', color='blue', label='Reference curve for negative curve type')
        if PosCurves_referance_curve is not None:
            plt.plot(X_IP_Window, PosCurves_referance_curve, '--', color='red', label='Reference curve for positive curve type')

        # Plotting the fitted decay curves and their corresponding referance curve(s). 
        plt.xlabel('Time [ms]')
        plt.ylabel('Fitted IP values [mV/V]')
        plt.title(f'Measurement ID: {sliced_rows["MeasID"].tolist()[0]}')
        plt.legend()
        plt.show()


        # >> Computing the shift value for each data point in the current electrode injection combination. 

        # Creating an empty 'shift' column to the dataframe df_IP_coordinates.
        df_IP_coordinates['shift'] = np.nan

        # Creating dictionaries to store optimal shift values for each DPID.
        optimal_shift_neg_dict = {}
        optimal_shift_pos_dict = {}
        
        # This function computes the value of the vertical shift the reference curve makes to end up with the smallest
        # rmsd between the reference curve and the respective fitted decay curve. 
        def optimal_shift(curve_type):
            curves = sliced_rows[sliced_rows["curve type"] == curve_type]
            if curves.empty:
                return None  
            
            # Selecting DPIDs of the sliced rows with the current curve type ('positive' or 'negative'). 
            unique_dpid_values_curves = curves["DPID"].unique()
            
            # Iterating over each DPID in the sliced rows. 
            for dpid in unique_dpid_values_curves:
            
                # Selecting the row corresponding to the current data point.
                DPID = curves["DPID"] == dpid
                current_row = curves[DPID]
                index = current_row.index.tolist()
                
                # Selecting only the values from columns starting with "IP#".
                ip_values_ref = current_row[ip_columns_ref].astype(float) 
                Y_IP_values_ref = ip_values_ref.loc[index[0]].tolist()
                
                # Function to compute the rmsd between the current fitted curve and the shifted reference curve. 
                def rmsd(shift, reference_curve):
                    shifted_curve = reference_curve + shift
                    return np.sqrt(np.mean((shifted_curve - Y_IP_values_ref) ** 2))
                
                # Function to compute the optimal vertical shift value for the referance curve to minimize the rmsd. 
                def compute_optimal_shift(reference_curve):
                    # Minimizing the rmsd between the current fitted curve and the shifted reference curve, using the Nelder-Mead method.
                    # The rmsd function is called and the initial shift guess is x0=0.
                    result = minimize(rmsd, x0=0, args=(reference_curve,), method='Nelder-Mead')
                    
                    # Defining optimal shift value. 
                    optimal_shift = result.x[0]
                    
                    return optimal_shift
                
                # Assigning the optimal shift values to dictionaries.
                if curve_type == 'negative':
                    if NegCurves_referance_curve is not None:
                        optimal_shift_neg_dict[dpid] = compute_optimal_shift(NegCurves_referance_curve)
                        
                elif curve_type == 'positive':    
                    if PosCurves_referance_curve is not None:
                        optimal_shift_pos_dict[dpid] = compute_optimal_shift(PosCurves_referance_curve)
            
                # Updating the 'shift' column for the curve types for the sliced rows.
                if curve_type == 'negative':        
                    for dpid, shift_value in optimal_shift_neg_dict.items():
                        curves.loc[curves['DPID'] == dpid, 'shift'] = shift_value        
                        
                elif curve_type == 'positive': 
                    for dpid, shift_value in optimal_shift_pos_dict.items():
                        curves.loc[curves['DPID'] == dpid, 'shift'] = shift_value
            
            # Returns a dataframe of the current curve type for the sliced rows with updated shift values. 
            return curves
        
        # Combines the dataframes of the positive and negative curve types for the current injection electrode combination.  
        sliced_rows_concatenated = pd.concat([optimal_shift("negative"), optimal_shift("positive")])
        
        # Appending the concatenated DataFrame to the a list. 
        concatenated_dfs.append(sliced_rows_concatenated)
            
        # Marking the current injection electrode combination as processed.
        processed_combinations.add(current_combination)


# Computing a dataframe with updated shift values for the whole dataset. 
df_IP_coordinates_shift = pd.concat(concatenated_dfs).sort_index() 

# Defining the DPID before the first filtering. 
DPID_BF1 = df_IP_coordinates_shift['DPID'].tolist() #BF1 = Before filter 1

# >> In the following section, the data points are devited and filtered for each curve type. 

# Defining the dataframe with negative and positive curve types respectivly. 
df_negative = df_IP_coordinates_shift[df_IP_coordinates_shift['curve type'] == 'negative']
df_positive = df_IP_coordinates_shift[df_IP_coordinates_shift['curve type'] == 'positive']

# Retreiving the upshift and downshift values ku and kd respectivly for negative and positive curve types respectivly.
ku_neg = df_negative[df_negative['shift'] > 0]['shift'] 
kd_neg = df_negative[df_negative['shift'] < 0]['shift']
ku_pos = df_positive[df_positive['shift'] > 0]['shift'] 
kd_pos = df_positive[df_positive['shift'] < 0]['shift']

# Calculating the standard deviation (std) for the upskift ku and the downshift kd respectively for the positive and negative 
# curve type for all current injections. 
# The std. is calculated for 'sample' data and not for 'population' data. 
ku_std_neg = ku_neg.std()
kd_std_neg = kd_neg.std()
ku_std_pos = ku_pos.std()
kd_std_pos = kd_pos.std()

# Assigning the total chargeability TCM to a dataframe with curve types. The dataframe dataOriginal_abmn and M_list (containing TCM values), 
# are arranged in the same order as the order in the dataOriginal file and is therefore comparable. 
dataOriginal_abmn['TCM'] = M_list

# Retrieving the rows for negative and positive curve types respectivly. 
TCM_negative = dataOriginal_abmn[dataOriginal_abmn['curve type'] == 'negative']
TCM_positive = dataOriginal_abmn[dataOriginal_abmn['curve type'] == 'positive']

# Computing the median value of the total chargeability for negative and positive curve types respectivly.  
TCM_median_negative = statistics.median(TCM_negative['TCM'].tolist())
TCM_median_positive = statistics.median(TCM_positive['TCM'].tolist())

# Identifies the dataset, for each curve type, as noisy, clean or general, respectivly, and defines their corresponding threshold values. 
# For negative curve types. 
if ku_std_neg*3 > TCM_median_negative*2:
    c_ku_neg = 1.5
    c_kd_neg = 1
elif ku_std_neg*3 < TCM_median_negative:
    c_ku_neg = 4
    c_kd_neg = 4
else:
    c_ku_neg = 3   
    c_kd_neg = 3
    
# For positive curve types. 
if ku_std_pos*3 > TCM_median_positive*2:
    c_ku_pos = 1.5
    c_kd_pos = 1
elif ku_std_pos*3 < TCM_median_positive:
    c_ku_pos = 4
    c_kd_pos = 4
else:
    c_ku_pos = 3   
    c_kd_pos = 3    

# Defining the conditions for filtering the negative and positive curve types. 
condition_ku_neg = df_IP_coordinates_shift['shift'] > c_ku_neg * ku_std_neg
condition_kd_neg = df_IP_coordinates_shift['shift'] < -(c_kd_neg * kd_std_neg)
condition_ku_pos = df_IP_coordinates_shift['shift'] > c_ku_pos * ku_std_pos
condition_kd_pos = df_IP_coordinates_shift['shift'] < -(c_kd_pos * kd_std_pos)

# Dropping the rows with outlier data points based on the conditions above. 
df_IP_coordinates_shift = df_IP_coordinates_shift.loc[~condition_ku_neg]
df_IP_coordinates_shift = df_IP_coordinates_shift.loc[~condition_kd_neg]
df_IP_coordinates_shift = df_IP_coordinates_shift.loc[~condition_ku_pos]
df_IP_coordinates_shift = df_IP_coordinates_shift.loc[~condition_kd_pos]


# Defining the DPID remaining after the filtering. 
DPID_AF1 = df_IP_coordinates_shift['DPID'].tolist() #AF1 = after filter 1

# Computing the DPID that were removed during the first filtering.
Filtered_DPID_F1= [DPID for DPID in DPID_BF1 if DPID not in DPID_AF1]
     

# ================================================= #
#                  Histogram filter                 #
# ================================================= #

# This filter identifies outlier data points as total chargeabilites without spatial correlation as the 
# histogram bins seperated from the main distribution by empty bins. 

# Defining the total number of measurements remaining in the data set after the first filtering. 
n =len(DPID_AF1)

# Defining the number of bins as proposed in [2]. 
nb = round(1+4.5*np.log10(n))

# Extracting the histogram values.
hist_values, bin_edges, _ = plt.hist(M_list, bins=nb)

# Creating a dataframe of the histogram values. 
data_histogram = {'bin_edges': bin_edges[:-1], 'hist_values': hist_values}
df_histogram = pd.DataFrame(data_histogram)

print("\nThe frequecy distibution of the total chargeability M is seen below. The histogram filter, will filter the data points seperated by the main distribution.\n")
print(df_histogram)

# Locating the index in the dataframe of the bin with the maximum value in the main distribution.  
index_max = df_histogram['hist_values'].idxmax()

# Locating the indecies in the histogram dataframe where bins are empty on the left and right side of the maximum, respectivly. 
zero_indices = df_histogram.index[df_histogram.eq(0).any(axis=1)]
zero_indices_left = [num for num in zero_indices if num < index_max]
zero_indices_right = [num for num in zero_indices if num > index_max]

# Defining the minimum and maximum total chargeability values as the values not seperated (by empty bins) from the main distibution. 
# Computing the index in the histogram dataframe for the minimum total chargeability value. 
if len(zero_indices_left) > 1:
    min_left = max(zero_indices_left)+1
elif len(zero_indices_left) == 1:
    min_left = zero_indices_left[0]+1
else:
    min_left = 0

# Computing the index in the histogram dataframe for the maximum total chargeability value. 
if len(zero_indices_right) > 1:
    max_right = min(zero_indices_right)-1
elif len(zero_indices_right) == 1:
    max_right = zero_indices_right[0]-1
else:
    # If the data is not seperated on the right side of the maximum value of the main distribution, the index variable is set to None. 
    max_right = None

# If some bins are seperated from the main distribution in the right side of the maximum, the max_right index is used to retrieve the 
# maximum TCM value. However, if the bins are not seperated on the right side of the maximum, the maximum TCM values is defined as the 
# maximum TCM from the data. This is due to the histogram not including the end value of the bin intervals. 
if max_right is not None:
    df_valid = df_histogram.loc[min_left:max_right]
    TCM_min_F2 = df_valid['bin_edges'].tolist()[0]
    TCM_max_F2 = df_valid['bin_edges'].tolist()[-1]
else:
    TCM_min_F2 = df_histogram['bin_edges'].loc[min_left]
    TCM_max_F2 = max(M_list)
    

# ================================================= #
#          Plotting and visualization data          #
# ================================================= #

# Plotting all the measured IP-decay curves with their fitted curves. 
plt.xlabel('Time [ms]')
plt.ylabel('IP values [mV/V]')
plt.title('Measured IP decay curves with their corresponding fitted decay curves')
plt.legend()
plt.show()


# Plotting the histogram. 
plt.hist(M_list, bins=nb, color='dodgerblue', alpha=0.7)
plt.title('Histogram of the total chargeabilities M')
plt.xlabel('Total chargeability M [mV/V]')
plt.ylabel('Frequency')
plt.show()


# Visualizing the total chargeability M and the goodness of fit (rmsd), between the fitted and measured curve, for the raw data. 
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, sharey=True, figsize=(8,5))

pg.physics.ert.showData(dataOriginal,"TCM",label="Chargeability M (mV/V)",ylabel="Pseudodepth",orientation="vertical",ax=ax1, cMin=TCM_min, cMax=TCM_max)
pg.physics.ert.showData(dataOriginal,"RMSD",label="Log$_{10}$ rmsd",ylabel="Pseudodepth",orientation="vertical",ax=ax2, cMin=rmsd_min, cMax=rmsd_max)

labels = [f"Raw data ({len(dataOriginal['ip7'])} data points)", "Goodness of fit"]
for ax, label in zip([ax1, ax2], labels):
    ax.set_xlim(profile_start, profile_end)
    ax.set_title(label)
    ax.set_yticklabels([])
ax.set_xlabel("Profile Distance (m)")


# Visualizing the total chargeability M and the goodness of fit after the first filtering. 
# Converting the list into matching type of dataOriginal['DPID'] 
# (Several attempts at converting dataOriginal['DPID'] into integers have failed)
Filtered_DPID_F1 = [float(x) for x in Filtered_DPID_F1]

# Copying the dataOrgiginal to apply the first filter to the .dat file for visualization. 
dataOriginal_F1 = dataOriginal.copy()
for i in range(len(Filtered_DPID_F1)):
    # Removing the data points that are not in the list of remaining datapoint after the first filtering. 
    dataOriginal_F1.remove(dataOriginal_F1['DPID'] == Filtered_DPID_F1[i])

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, sharey=True, figsize=(8,5))

pg.physics.ert.showData(dataOriginal_F1,"TCM",label="Opladelighed M (mV/V)",ylabel="Pseudodybde",orientation="vertical",ax=ax1, cMin=TCM_min, cMax=TCM_max)
pg.physics.ert.showData(dataOriginal_F1,"RMSD",label="Log$_{10}$ rmsd",ylabel="Pseudodybde",orientation="vertical",ax=ax2, cMin=rmsd_min, cMax=rmsd_max)

labels = [f"Efter filtreringstrin 1 ({len(dataOriginal_F1['ip7'])} data punkter)", "Goodness of fit efter filtreringstrin 1"]
for ax, label in zip([ax1, ax2], labels):
    ax.set_xlim(profile_start, profile_end)
    ax.set_title(label)
    ax.set_yticklabels([])
ax.set_xlabel("Profilafstand (m)")


# Vizualizing the total chargeability M and the goodness of fit after the second filtering. 
# Copying the dataOrgiginal_F1 to apply the second filter for visualization
dataOriginal_F2 = dataOriginal_F1.copy()

# Filtering data points based on the minimum and maximum TCM values computed in the histogram filter. 
dataOriginal_F2.remove(dataOriginal_F2['TCM'] > TCM_max_F2)
dataOriginal_F2.remove(dataOriginal_F2['TCM'] < TCM_min_F2)

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, sharey=True, figsize=(8,5))

pg.physics.ert.showData(dataOriginal_F2,"TCM",label="Opladelighed M (mV/V)",ylabel="Pseudodybde",orientation="vertical",ax=ax1, cMin=TCM_min, cMax=TCM_max)
pg.physics.ert.showData(dataOriginal_F2,"RMSD",label="Log$_{10}$ rmsd",ylabel="Pseudodybde",orientation="vertical",ax=ax2, cMin=rmsd_min, cMax=rmsd_max)

labels = [f"Efter filtreringstrin 2 ({len(dataOriginal_F2['TCM'])} data punkter)", "Goodness of fit efter filtreringstrin 2"]
for ax, label in zip([ax1, ax2], labels):
    ax.set_xlim(profile_start, profile_end)
    ax.set_title(label)
    ax.set_yticklabels([])
ax.set_xlabel("Profil afstand (m)")


# Function to plot the data
def plot_data(ax, data, param, label, cMin, cMax, profile_start, profile_end):
    pg.physics.ert.showData(data, param, label=label, ylabel="Pseudodybde", orientation="vertical", ax=ax, cMin=cMin, cMax=cMax)
    ax.set_xlim(profile_start, profile_end)
    ax.set_yticklabels([])

# Create the main figure with 3 rows and 2 columns
fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(13, 7))

# Raw data plots
plot_data(axs[0, 0], dataOriginal, "TCM", "Opladelighed M (mV/V)", TCM_min, TCM_max, profile_start, profile_end)
axs[0, 0].set_title(f"Rå data ({len(dataOriginal['ip7'])} data punkter)")
plot_data(axs[0, 1], dataOriginal, "RMSD", "Log$_{10}$ rmsd", rmsd_min, rmsd_max, profile_start, profile_end)
axs[0, 1].set_title("Goodness of fit for rå data")

# Filtering step 1 plots
Filtered_DPID_F1 = [float(x) for x in Filtered_DPID_F1]
dataOriginal_F1 = dataOriginal.copy()
for i in range(len(Filtered_DPID_F1)):
    dataOriginal_F1.remove(dataOriginal_F1['DPID'] == Filtered_DPID_F1[i])

plot_data(axs[1, 0], dataOriginal_F1, "TCM", "Opladelighed M (mV/V)", TCM_min, TCM_max, profile_start, profile_end)
axs[1, 0].set_title(f"Efter filtreringstrin 1 ({len(dataOriginal_F1['ip7'])} data punkter)")
plot_data(axs[1, 1], dataOriginal_F1, "RMSD", "Log$_{10}$ rmsd", rmsd_min, rmsd_max, profile_start, profile_end)
axs[1, 1].set_title("Goodness of fit efter filtreringstrin 1")

# Filtering step 2 plots
dataOriginal_F2 = dataOriginal_F1.copy()
dataOriginal_F2.remove(dataOriginal_F2['TCM'] > TCM_max_F2)
dataOriginal_F2.remove(dataOriginal_F2['TCM'] < TCM_min_F2)

plot_data(axs[2, 0], dataOriginal_F2, "TCM", "Opladelighed M (mV/V)", TCM_min, TCM_max, profile_start, profile_end)
axs[2, 0].set_title(f"Efter filtreringstrin 2 ({len(dataOriginal_F2['TCM'])} data punkter)")
plot_data(axs[2, 1], dataOriginal_F2, "RMSD", "Log$_{10}$ rmsd", rmsd_min, rmsd_max, profile_start, profile_end)
axs[2, 1].set_title("Goodness of fit efter filtreringstrin 2")

# Set common labels
for ax in axs[:, 0]:
    ax.set_ylabel("Pseudodybde")
for ax in axs[2, :]:
    ax.set_xlabel("Profilafstand (m)")

plt.tight_layout()
plt.show()


# ================================================= #
#     Exporting the filtered data to a DAT file     #
# ================================================= #

# In this section the filtered data is exported to a .dat file, which is compatable with Res2Dinv. 

# start_ip_window and end_ip_window is the range of the included IP-windows. E.g. IP-window 2-9. 
# delay_integration_time is a list of the delay and integration time, of the exported IP-window. e.g. delay_integration_time = [0.01, 0.04, 0.06, 0.14, 0.26, 0.5, 0.96, 2, 2]

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
    # %% write res2Dinv file
    with open(filename, 'w') as fi:
        fi.write(arrayName+'\n')
        fi.write(str(np.round(min(dist2), decimals=2))+'\n')
        fi.write(str(ar_idfy)+'\n')
        fi.write('0\n')
        fi.write('Type of measurement (0=app.resistivity,1=resistance) \n')
        fi.write(datType+'\n')
        fi.write(str(len(data('r')))+'\n')
        fi.write('2'+'\n')
        if data.allNonZero('ip'):
            fi.write('1\n')
            fi.write('Chargeability\n')
            fi.write('mV/V\n')
            delay_integration_time = ' '.join(map(str, delay_integration_time))
            fi.write(delay_integration_time)  # delay/integration time
            fi.write("\n")
            if data.allNonZero('r'):
                lines = []
                for oo in range(len(data('a'))):
                    ip_values = [str(data(f'ip{j}')[oo]) for j in range(start_ip_window, end_ip_window + 1)]
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
                           sep.join(ip_values)
                    lines.append(line)
            else:
                lines = []
                for oo in range(len(data('a'))):
                    ip_values = [str(data(f'ip{j}')[oo]) for j in range(start_ip_window, end_ip_window + 1)]
                    line = '4' + sep + \
                           str(x[int(data('a')[oo])]) + sep + \
                           str(y[int(data('a')[oo])]) + sep + \
                           str(x[int(data('b')[oo])]) + sep + \
                           str(y[int(data('b')[oo])]) + sep + \
                           str(x[int(data('m')[oo])]) + sep + \
                           str(y[int(data('m')[oo])]) + sep + \
                           str(x[int(data('n')[oo])]) + sep + \
                           str(y[int(data('n')[oo])]) + sep + \
                           sep.join(ip_values)
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
export_if_not_exists(dataOriginal_F2, start_ip_window, end_ip_window, [len(delay_integration_time)] + delay_integration_time + current_ON_OFF, file_location)


#----------------------------- References ------------------------------------#
'''
[1] Guideline GEO, personal communication, 2024

[2] Adrian Flores Orozco et al. ‘Decay curve analysis for data error quantification in time-domain
induced polarization imaging’. In: Geophysics 83 (2 Mar. 2018), E75–E86. issn: 19422156.
doi: 10.1190/geo2016-0714.1 

'''