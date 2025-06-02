import pandas as pd # type: ignore
import numpy as np
import os

# Define base directory where the 30 folders (1 to 30 runs) exist
# Path to the base directory containing 30 subfolders (1 to 30), each with a CSV file.
# BASE_DIR = "/Users/tanjina/Desktop/30runs-new/500Mbit-LAN/cheetah-resnet50-client-output"
# BASE_DIR = "/Users/tanjina/Desktop/30runs-new/500Mbit-LAN/NEW/cheetah-resnet50-client-output"
BASE_DIR = "/Users/tanjina/Desktop/New-readings/output-with-sleep-power-reading/client/3s"

# Path to the final Output CSV file name "per_layer_energy_consumption_30_runs.csv", where the calculated energy measurements per layer will be stored
# OUTPUT_FILE = "/Users/tanjina/Desktop/30runs-new/500Mbit-LAN/client_output/per_layer_energy_consumption_30_runs.csv" # Comment this when want to use transpose
# OUTPUT_FILE = "/Users/tanjina/Desktop/30runs-new/500Mbit-LAN/NEW/client_output/per_layer_energy_consumption_30_runs.csv"
# OUTPUT_FILE = "/Users/tanjina/Desktop/30runs-new/500Mbit-LAN/client_output/per_layer_energy_consumption_30_runs_transposed.csv" # Comment out this when want to use transpose
# OUTPUT_FILE = "/Users/tanjina/Desktop/30runs-new/500Mbit-LAN/NEW/client_output/per_layer_energy_consumption_30_runs_transposed.csv"

# OUTPUT_FILE = "/Users/tanjina/Desktop/New-readings/output-with-sleep-power-reading/client/client_output/per_layer_energy_consumption_sleep_3s_30_runs.csv"
OUTPUT_FILE = "/Users/tanjina/Desktop/New-readings/output-with-sleep-power-reading/client/client_output/per_layer_energy_consumption_sleep_3s_30_runs_transposed.csv"
# Total number of run
NUM_RUNS = 30

"""
Function to compute energy consumption per layer from an input CSV file
"""
def calculate_energy_consumption_per_layer(file_path, timestamp_col, power_col, layer_col):
    """
    Calculate the energy consumption using the formula:
    E = Summation ((Pi + Pi+1)/2  * (ti+1 - ti) * 1e-9 )  # Convert to joules
    for each layer_number separately.

    Parameters:
        file_path (str): Path to the CSV file containing the dataset.
        timestamp_col (str): Name of the column containing timestamps of the power usage value.
        power_col (str): Name of the column containing power usage values.
        layer_col (str): Name of the column containing layer numbers.

    Returns:
        layer_energy_results: Dictionary containing total energy consumption for each layer.
    """

    try:
        # Load the dataset
        data = pd.read_csv(file_path)
        # Convert the timestamp column from milliseconds to datetime format
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], unit='ms')
        # Group data by layer
        grouped = data.groupby(layer_col)
        # Initialize a list to store energy contributions for each layer
        # Stores the result in layer_energy_results dictionary, where keys are layers and values are energy consumption (Joules).
        layer_energy_results = {}

         # Process each group (layer) separately
        for layer, group in grouped:
            timestamps = group[timestamp_col].values # timestamps as a list
            power_values = group[power_col].values # power_values as a list

            # Initialize total energy consumption for the current layer
            total_energy = 0
            
            # Iterate the rows in the input dataset except the last index of power_values
            for i in range(len(power_values) - 1):
                # Time difference as a numpy timedelta64
                time_diff = (timestamps[i + 1] - timestamps[i])

                # Convert time difference from timedelta64 to milliseconds (numerical value)
                # time_diff_ms = time_diff.astype('timedelta64[ms]').item() # Convert the time delta in milliseconds, .item(), extracts the scalar value from the numpy.timedelta64, turning it into a standard integer or float (in this case, milliseconds).
                # time_diff_ms = time_diff.total_seconds() * 1000  # in milliseconds
                time_diff_ms = time_diff / np.timedelta64(1, 'ms')


                # Average power in microwatts between two consecutive readings
                avg_power = (power_values[i] + power_values[i + 1]) / 2

                # Energy contribution for this interval in Joules
                energy_contribution = avg_power * time_diff_ms * 1e-9  # Convert to joules
                #print(f"Layer {layer}, Interval {i} - Energy Contribution: {energy_contribution} Joules")

                # Add to total energy for this layer
                total_energy += energy_contribution

            # Store total energy consumption in Joules for the current layer    
            layer_energy_results[layer] = total_energy
        # print(f"Energy data for {file_path}: {layer_energy_results}") # For debugging, comment out this
        return layer_energy_results
    
    except Exception as e:
        print(f"{e}: Error in processing {file_path}")
        return {}

def process_all_runs(base_dir, num_runs):
    """
    Processes all 30 runs and computes energy consumption for each convolutional layer.
    """
    # dictionary to store the energy measurements
    layer_energy_dict = {}
    
    # Iterate through each folder from the base dir, ranges 1 - 30
    for run in range(1, num_runs + 1):
        current_csv_file_path = os.path.join(base_dir, str(run), f"conv_output_run_{run}.csv") # ex: 1/conv_output_run_1.csv
        
        if not os.path.exists(current_csv_file_path):
            print(f"Warning: {current_csv_file_path} not found, skipping...")
            continue
        
        # Process energy measurement per run
        # column names for timestamp, avg_power, and layer number
        energy_data_per_run = calculate_energy_consumption_per_layer(current_csv_file_path, "timestamp_power_reading", "avg_power_usage_mcW", "layer_number")
        
        if energy_data_per_run:  # Only proceed if we got some energy data
            for layer, energy in energy_data_per_run.items():
                if layer not in layer_energy_dict:
                    layer_energy_dict[layer] = []
                layer_energy_dict[layer].append(energy)


    # Check if we have any energy data
    if not layer_energy_dict:
        print("No energy data processed. Check your input files.")
        return pd.DataFrame()  # Return an empty DataFrame if no data was processed
    # Need to convert the dictionary into a new dataframe where:
    # - Each row now represnts conv_layer# ranges: 1 to 53 (for ResNet50)
    # - Each column now represnts runs# ranges ( run_1 to run_30)   
    output_data = pd.DataFrame.from_dict(layer_energy_dict, orient='index').sort_index()
    output_data.index.name = "Conv_layer#"
    # rename the columns to "run_1", ....."run_30"
    output_data.columns = [f"run_{i}" for i in range (1, num_runs + 1)]

    # Round the values to 4 decimal places
    output_data = output_data.round(4)

    print(f"Processed data for {len(output_data)} layers.")
    # return output_data # Rows: conv_layer_num#  (1 to 53), Columns: run_num# (run_1 to run_30) # Comment this when want to use transpose_data_frame
    


    # # Transpose the data frame to swap row and column
    # # - Each row will contain run_num# ranges: 1 to 30
    # # - Each column will contain layer_number# ranges: 1 to 53 (for ResNet50)

    return transpose_data_frame(output_data) # Comment out this code when want to return transpose_data_frame

def transpose_data_frame(data_frame):
    # Transpose the data frame to swap row and column
    # - Each row will contain run_num# ranges: 1 to 30
    # - Each column will contain layer_number# ranges: 1 to 53 (for ResNet50)
    output_data_transposed = data_frame.T

    # rename the columns to "conv_layer_1", "conv_layer_2", ...
    output_data_transposed.columns = [f"conv_layer_{conv_layer}" for conv_layer in range (1,54)]
    # Rename the index (Run#) to 1, 2, ..., 30 instead of "run_1", "run_2", ...
    output_data_transposed.index = range(1, 31)  # Each row now represents run_num# ranges: 1 to 30
    # Rename the index column
    output_data_transposed.index.name = "Run"

    print(f"Transposed per layer energy consumption are saved at: {OUTPUT_FILE} file")

    return output_data_transposed # Rows: run_num# (1 to 30), Columns: conv_layer_num# (conv_layer_1 to conv_layer_53)

def main():
    print("Processing energy consumption for 30 runs...")

    # Process all CSV files and compute the layer-wise energy consumption for each run
    energy_summary_df = process_all_runs(BASE_DIR, NUM_RUNS)

    if energy_summary_df.empty:
        print("No energy consumption data was processed.")
    else:
        # Save results to a CSV file
        energy_summary_df.to_csv(OUTPUT_FILE)
        print(f"Energy measurements saved to {OUTPUT_FILE}")

    
    # print(f"Energy measurements saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
