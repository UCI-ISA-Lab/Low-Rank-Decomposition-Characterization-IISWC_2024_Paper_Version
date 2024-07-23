import os
import json
import pandas as pd

benchmarks = "arc_easy,arc_challenge,hellaswag,mmlu,truthfulqa,winogrande"
print_name_benchmarks = "ARC Easy,ARC Challenge,HellaSwag,MMLU,TruthfulQA,WinoGrande"


decomposition_config = {
    "layers_to_decompose": [5],
    "matrices_to_decompose": [1, 1, 1, 1, 1, 1, 1],
    "rank": [1, 1],
    "collect_power": 1
}

#Accuracy Impact
matrices_to_decompose = [ [1, 1, 1, 1, 1, 1, 1] ]
ranks = [ [1, 1] ]
# Percentage Decomposition: 6, 9, 15, 21, 33, 48, 60, 75, 84 ,96 
# Num of Layers Decomposed:  2, 3, 5,  7,  11, 16, 20, 25, 28 ,32
layers_to_decompose =   [ [] ] \
                        + [ [2, 29] ] \
                        + [ [2, 17, 31] ] \
                        + [ [i for i in range(2, 32, 6)] ] \
                        + [ [i for i in range(4, 32, 4)] ] \
                        + [ [i for i in range(2, 32, 3)] + [31]] \
                        + [ [i for i in range(0, 32, 2)] ] \
                        + [ [i for i in range(1, 10, 2)] + [i for i in range(10, 19, 1)] + [i for i in range(20, 32, 2)] ] \
                        + [ [i for i in range(1, 10, 2)] + [i for i in range(10, 30, 1)] ] \
                        + [ [i for i in range(0, 8, 2)] + [i for i in range(8, 32, 1)] ] \
                        + [ [i for i in range(32)] ]

#variables
inference_time = None
dataset_size = None
decomposition_percentage = None
output = None
df_list = []

# for layer, matrix, file_name in zip(layers_to_decompose, matrices_to_decompose, file_names):
for layer in layers_to_decompose:
    for matrix in matrices_to_decompose:
        for rank in ranks:
            decomposition_config["layers_to_decompose"] = layer
            decomposition_config["matrices_to_decompose"] = matrix
            decomposition_config["rank"] = rank
            with open("decomposition_config.json", "w") as file:
                json_object = json.dumps(decomposition_config)
                file.write(json_object)
            for benchmark, benchmark_name in zip(benchmarks.split(","), print_name_benchmarks.split(",")):
                os.system(f"rm -rf power.log output.jsonl terminal_output.log")
                os.system(f"./run.sh {benchmark}")
                with open("temp_values.json", "r") as file:
                    temp = json.load(file)
                    decomposition_percentage = temp["decomposition_percentage"]
                results = {
                    "Layers Decomposed": [layer],
                    "Matrices Decomposed": [matrix],
                    "Rank": rank[0],
                    "Decomposition Percentage": decomposition_percentage,
                    "Inference Time": inference_time,
                    "Dataset Size": dataset_size,
                    "Metric": "Accuracy"
                }
                with open("output.jsonl", "r") as file:
                    output = json.load(file)
                results[benchmark_name] = output["results"][benchmark]["acc,none"]
                df = pd.DataFrame(results, index=[0])
                df_list.append(df)
                os.system(f"mv power.log power_output/power_logs/{benchmark}_{round(decomposition_percentage, 2)}%_decomp_{len(layer)}_decomp.log")
                os.system(f"mv output.jsonl power_output/jsonl_output/{benchmark}_{round(decomposition_percentage, 2)}%_decomp_{len(layer)}_decomp.jsonl")
                os.system(f"mv terminal_output.log power_output/terminal_logs/{benchmark}_{round(decomposition_percentage, 2)}%_decomp_{len(layer)}_decomp.log")
            # layer_difference = layer_difference - 1
            # os.system(f"mv output.jsonl output_dir/matrix_sensitivty_{len(layer)}_layers_{decomposition_percentage}%decomposition_{sum(matrix)}_matrices.jsonl")
            # os.system(f"mv terminal_output.log output_logs/rank_sensitivty_{len(layer)}_layers_decomposed_{decomposition_percentage}%decomposition_rank_{rank[0]}.log")

final_df = pd.concat(df_list, ignore_index=True)
print(final_df)
final_df.to_csv("output_dir/accuracy_impact_data.csv", index=False)