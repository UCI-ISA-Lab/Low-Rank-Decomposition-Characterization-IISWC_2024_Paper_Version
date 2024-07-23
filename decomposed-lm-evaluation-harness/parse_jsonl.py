import json
import re
import os
import pandas as pd

files = os.listdir("old_output_dir")
files = [ file for file in files if "matrix_sensitivity" in file ]
all_layers_decomposed_files = [ file for file in files if "all_layers_decomposed" in file ]
all_layers_decomposed_files = sorted(all_layers_decomposed_files)
single_layer_decomposed_files = [ file for file in files if "single_layer_decomposed" in file ]
single_layer_decomposed_files = sorted(single_layer_decomposed_files)

all_data = {
    "columns": ["arc_challenge", "arc_easy", "openbookqa", "piqa", "winogrande"]
}

count = 0
for file in all_layers_decomposed_files:
    f = os.path.join("old_output_dir", file)
    with open(f, "r") as json_file:
        data = json.load(json_file)
        all_data["all_layers_matrix_" + str(count)] = []
        for benchmark in all_data["columns"]:
            all_data["all_layers_matrix_" + str(count)].append(data["results"][benchmark]["acc,none"])
    count = count + 1

count = 0
for file in single_layer_decomposed_files:
    f = os.path.join("old_output_dir", file)
    with open(f, "r") as json_file:
        data = json.load(json_file)
        all_data["single_layer_matrix_" + str(count)] = []
        for benchmark in all_data["columns"]:
            all_data["single_layer_matrix_" + str(count)].append(data["results"][benchmark]["acc,none"])
    count = count + 1
    
df = pd.DataFrame.from_dict(all_data, orient="index")
df.to_csv("old_output_dir/matrix_senitivity_sweep.csv")