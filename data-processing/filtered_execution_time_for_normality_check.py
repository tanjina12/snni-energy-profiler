import pandas as pd
import os

# Path to the base directory
# base_dir_path = "/Users/tanjina/Desktop/30runs-new/500Mbit-LAN/cheetah-resnet50-client-output"
# base_dir_path = "/Users/tanjina/Desktop/30runs-new/500Mbit-LAN/NEW/cheetah-resnet50-client-output"
# base_dir_path = "/Users/tanjina/Desktop/30runs-new/500Mbit-LAN/NEW/without-sleep-without-power-reading/client"
base_dir_path = "/Users/tanjina/Desktop/New-readings/output-with-sleep-power-reading/client/3s"

# to store the execution time per run
execution_time = {}

# Iterate through each folder from the base dir, ranges 1 - 30
for run in range(1, 31):
    file_path  = os.path.join(base_dir_path, str(run), f"conv_output_run_{run}.csv")

    if os.path.exists(file_path):
        # read the csv fie
        data = pd.read_csv(file_path)

        # drop duplicate execution_time_ms entries for each layer_number and keep only the first occurance
        filtered_data = data.drop_duplicates(subset=['layer_number'], keep = 'first')

        # store the execution times with their corresponding layer number
        #execution_time[run] = filtered_data.set_index("layer_number")["execution_time_ms"].to_dict()
        execution_time[run] = {f"conv_layer_{layer}": eTime for layer, eTime in filtered_data.set_index("layer_number")["execution_time_ms"].to_dict().items()}
    else:
        print(f"{file_path}: No such file found!")    

# Need to convert the dictionary into a new dataframe where:
# - Each row will contain run_num# ranges: 1 to 30
# - Each column will contain layer_number# ranges: 1 to 53 (for ResNet50)
output_data = pd.DataFrame.from_dict(execution_time, orient='index').sort_index()

# Rename the index column
output_data.index.name = "Run"

# path to the output csv file
# output_path = os.path.abspath("/Users/tanjina/Desktop/30runs-new/500Mbit-LAN/client_output/per_layer_execution_time_30_runs.csv")
# output_path = os.path.abspath("/Users/tanjina/Desktop/30runs-new/500Mbit-LAN/NEW/client_output/per_layer_execution_time_30_runs.csv")
# output_path = os.path.abspath("/Users/tanjina/Desktop/30runs-new/500Mbit-LAN/NEW/without-sleep-without-power-reading/client_output/per_layer_benchmark_execution_time_30_runs.csv")
output_path = os.path.abspath("/Users/tanjina/Desktop/New-readings/output-with-sleep-power-reading/client/client_output/per_layer_execution_time_sleep_3s_30_runs.csv")
# Write the final dataframe to a new csv file
output_data.to_csv(output_path)

print(f"Filtered per layer executon times are saved at: {output_path} file")

# Write the final dataframe to a new csv file
#output_data.to_csv("filtered_execution_times.csv")

#print("Filtered executon times are saved to the CSV file")

# Transpose the data frame to swap row and column
# - Each row now represnts conv_layer# ranges (conv_layer_1 to conv_layer_53)
# - Each column now represnts runs# ranges ( run_1 to run_30)
output_data_transposed = output_data.T


# rename the columns to "run_1", ....."run_30"
output_data_transposed.columns = [f"run_{i}" for i in range (1,31)]
# Rename the index (Conv_layer#) to 1, 2, ..., 53 instead of "conv_layer_1", "conv_layer_2", ...
output_data_transposed.index = range(1, 54)  # Each row now represnts Conv_layer# ranges: 1 to 53 (for ResNet50)
# Rename the index column
output_data_transposed.index.name = "Conv_layer#"

# path to the transposed output csv file along with new column names
# output_transposed_path = os.path.abspath("/Users/tanjina/Desktop/30runs-new/500Mbit-LAN/client_output/per_layer_execution_time_30_runs_transposed.csv")
# output_transposed_path = os.path.abspath("/Users/tanjina/Desktop/30runs-new/500Mbit-LAN/NEW/client_output/per_layer_execution_time_30_runs_transposed.csv")
# output_transposed_path = os.path.abspath("/Users/tanjina/Desktop/30runs-new/500Mbit-LAN/NEW/without-sleep-without-power-reading/client_output/per_layer_benchmark_execution_time_30_runs_transposed.csv")
output_transposed_path = os.path.abspath("/Users/tanjina/Desktop/New-readings/output-with-sleep-power-reading/client/client_output/per_layer_execution_time_sleep_3s_30_runs_transposed.csv")
# Write the final dataframe to a new csv file
output_data_transposed.to_csv(output_transposed_path)

#print(output_data_transposed.head())

print(f"Transposed filtered per layer executon times are saved at: {output_transposed_path} file")
