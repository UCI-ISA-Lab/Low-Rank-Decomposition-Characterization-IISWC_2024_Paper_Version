import os
import json
import pandas as pd
from tqdm import tqdm

# files = os.listdir("output_dir")
# files = sorted(files)
# for file in files:
#     f = os.path.join("output_dir", file)
#     with open(f, "r") as output_file:
#         data = json.load(output_file)
#         print(data["results"]["piqa"]["acc,none"])
# exit()


# benchmarks = "arc_easy,arc_challenge,hellaswag,mmlu,truthfulqa,winogrande,gsm8k"
# lite_benchmarks = "arc_easy,arc_challenge,hellaswag,winogrande"
# heavy_benchmarks = "mmlu,truthfulqa,gsm8k"

benchmarks = "arc_easy,arc_challenge,hellaswag,mmlu,truthfulqa,winogrande,gsm8k"
print_name_benchmarks = "ARC Easy,ARC Challenge,HellaSwag,MMLU,TruthfulQA,WinoGrande,GSM8K"

decomposition_config = {
    "layers_to_decompose": [5], #0 - 31
    "matrices_to_decompose": [1, 1, 1, 1, 1, 1, 1], # 7 matrices per layer
    "rank": [1, 1]
}

#Rank Sweep
# ranks = [ [500, 500], [250, 250], [1, 1] ]
# layers_to_decompose = [[i for i in range(3, 32, 2)]] + [[i for i in range(3, 32, 3)]] + [[i for i in range(3, 32, 5)]]
# matrices_to_decompose = [[1, 1, 1, 1, 1, 1, 1]]

#Layer Sweep
# ranks = [ [500, 500], [250, 250], [1, 1] ]
# layers_to_decompose = [[i for i in range(3, 32, 2)]] + [[i for i in range(3, 32, 3)]] + [[i for i in range(3, 32, 5)]]
# matrices_to_decompose = [[1, 1, 1, 1, 1, 1, 1]]

#Matrix Sweep
# ranks = [ [1, 1] ]
# layers_to_decompose = []
# matrices_to_decompose = []
# for i in range(7): #all layers each matrix - [8%, 8, 8, 8, 21, 21, 21]
#     matrix = [0, 0, 0, 0, 0, 0, 0]
#     layer = [i for i in range(32)]
#     matrix[i] = 1
#     matrices_to_decompose.append(matrix)
#     layers_to_decompose.append(layer)
# #3% Decomposition
# layers_to_decompose.append([3, 15, 27])
# matrices_to_decompose.append([1, 1, 1, 1, 1, 1, 1])
# #21% Decomposition
# layers_to_decompose.append([3, 7, 11, 15, 21, 25, 29])
# matrices_to_decompose.append([1, 1, 1, 1, 1, 1, 1])
# for i in range(7): #single layer each matrix
#     matrix = [0, 0, 0, 0, 0, 0, 0]
#     layer = [3]
#     matrix[i] = 1
#     matrices_to_decompose.append(matrix)
#     layers_to_decompose.append(layer)
# layers_to_decompose.append([3])
# matrices_to_decompose.append([1, 1, 1, 1, 1, 1, 1])
# layers_to_decompose.append([3])
# matrices_to_decompose.append([1, 1, 1, 1, 0, 0, 0])
# layers_to_decompose.append([3])
# matrices_to_decompose.append([0, 0, 0, 0, 1, 1, 1])
# file_names = []
# for i in range(7):
#     name = f"matrix_sensitivity_32_Layers_matrix_{i}"
#     file_names.append(name)
# file_names.append(f"matrix_sensitivty_3_Layers_9%_Decomp_all_matrices")
# file_names.append(f"matrix_sensitivty_7_Layers_21%_Decomp_all_matrices")
# for i in range(10):
#     if(i < 7):
#         name = f"matrix_sensitivity_1_Layer_matrix_{i}"
#     elif (i == 7):
#         name = f"matrix_sensitivity_1_Layer_all_matrices"
#     elif (i == 8):
#         name = f"matrix_sensitivity_1_Layer_self_attn_matrices"
#     elif (i == 9):
#         name = f"matrix_sensitivity_1_Layer_MLP_matrices"
#     file_names.append(name)
    
#Layer Density Sweep
# ranks = [ [1, 1]]
# matrices_to_decompose = [ [1, 1, 1, 1, 1, 1, 1] ]
# layers_to_decompose = []
# starting_layer = 2
# layer_difference = 6
# curr_layer = starting_layer
# while (layer_difference > 0):
#     layer = []
#     for i in range(5):
#         curr_layer = starting_layer + i * layer_difference
#         layer.append(curr_layer)
#     layers_to_decompose.append(layer)
#     layer_difference = layer_difference - 1

#Accuracy Impact
# matrices_to_decompose = [ [1, 1, 1, 1, 1, 1, 1] ]
# ranks = [ [1, 1] ]
# # Percentage Decomposition: 6, 9, 15, 21, 33, 48, 60, 75, 84 ,96 
# #Num of Layers Decomposed:  2, 3, 5,  7,  11, 16, 20, 25, 28 ,32
# layers_to_decompose =     [ [2, 29] ] \
#                         + [ [2, 17, 31] ] \
#                         + [ [i for i in range(2, 32, 6)] ] \
#                         + [ [i for i in range(4, 32, 4)] ] \
#                         + [ [i for i in range(2, 32, 3)] + [31]] \
#                         + [ [i for i in range(0, 32, 2)] ] \
#                         + [ [i for i in range(1, 10, 2)] + [i for i in range(10, 19, 1)] + [i for i in range(20, 32, 2)] ] \
#                         + [ [i for i in range(1, 10, 2)] + [i for i in range(10, 30, 1)] ] \
#                         + [ [i for i in range(0, 8, 2)] + [i for i in range(8, 32, 1)] ] \
#                         + [ [i for i in range(32)] ]

#Fine Tuning
matrices_to_decompose = [ [1,1,1,1,1,1,1] ]
# matrices_to_decompose = [ [1, 1, 1, 1, 1, 1, 1] ]
ranks = [ [1, 1] ]
# layers_to_decompose = [ [i for i in range(2, 32, 6)] ]
# layers_to_decompose = [ [i for i in range(32) ] ]
# layers_to_decompose = [ [0] ]
# layers_to_decompose = [ [3, 9, 15, 21, 27] ]
layers_to_decompose = [ [2, 29] ]
# benchmarks = "arc_easy,arc_challenge,hellaswag,mmlu,truthfulqa,winogrande"
benchmarks = "arc_easy"

#variables
inference_time = None
dataset_size = None
decomposition_percentage = None
output = None
df_list = []

# layer_difference = 6
for layer in tqdm(layers_to_decompose):
    for matrix in matrices_to_decompose:
# for layer, matrix, file_name in zip(layers_to_decompose, matrices_to_decompose, file_names):
        for rank in ranks:
            # if(os.path.isfile(f"output_dir/{file_name}.jsonl")):
            #     with open(f"output_dir/{file_name}.jsonl", "r") as file:
            #         output = json.load(file)
            #     results = {
            #         "Layers Decomposed": [output["layers_decomposed"]],
            #         "Matrices Decomposed": [output["matrices_decomposed"]],
            #         "Rank": [output["decomposition_rank"]],
            #         "Decomposition Percentage": [output["decomposition_percentage"]],
            #         "Inference Time": [output["inference_time"]],
            #         "Dataset Size": [output["dataset_size"]],
            #         "Metric": "Accuracy"
            #     }
            #     for benchmark, benchmark_name in zip(benchmarks.split(","), print_name_benchmarks.split(",")):
            #         results[benchmark_name] = output["results"][benchmark]["acc,none"]
            #     df = pd.DataFrame(results, index=[0])
            #     df_list.append(df)
            # else:
            decomposition_config["layers_to_decompose"] = layer
            decomposition_config["matrices_to_decompose"] = matrix
            decomposition_config["rank"] = rank
            with open("decomposition_config.json", "w") as file:
                json_object = json.dumps(decomposition_config)
                file.write(json_object)
            os.system(f"./run.sh {benchmarks}")
            with open("temp_values.json", "r") as file:
                temp = json.load(file)
                inference_time = temp["inference_time"]
                dataset_size = temp["dataset_size"]
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
            for benchmark, benchmark_name in zip(benchmarks.split(","), print_name_benchmarks.split(",")):
                if(benchmark == "gsm8k"):
                    results[benchmark_name] = output["results"][benchmark]["exact_match,get-answer"]
                    results["Metric"] = "Exact Match"
                    if(output["results"][benchmark]["exact_match,get-answer"] == 0):
                        benchmarks = "arc_easy,arc_challenge,hellaswag,mmlu,truthfulqa,winogrande"
                        print_name_benchmarks = "ARC Easy,ARC Challenge,HellaSwag,MMLU,TruthfulQA,WinoGrande"
                else:
                    results[benchmark_name] = output["results"][benchmark]["acc,none"]
            df = pd.DataFrame(results, index=[0])
            df_list.append(df)
            
            #Save output json (Not important)
            os.system(f"mv output.jsonl output_dir/testing_finetuning_output_{len(layer)}_layers_decomposed.jsonl")
            os.system(f"mv terminal_output.log output_logs/testing_finetuning_output_{len(layer)}_layers_decomposed.log")
            # os.system(f"mv output.jsonl output_dir/finetuning_output_{len(layer)}_layers_decomposed.jsonl")
            # os.system(f"mv terminal_output.log output_logs/finetuning_output_{len(layer)}_layers_decomposed.log")
            # layer_difference = layer_difference - 1
            # os.system(f"mv output.jsonl output_dir/matrix_sensitivty_{len(layer)}_layers_{decomposition_percentage}%decomposition_{sum(matrix)}_matrices.jsonl")
            # os.system(f"mv terminal_output.log output_logs/rank_sensitivty_{len(layer)}_layers_decomposed_{decomposition_percentage}%decomposition_rank_{rank[0]}.log")

#Change output directory path
final_df = pd.concat(df_list, ignore_index=True)
print(final_df)
final_df.to_csv(f"output_dir/testing_finetuning_data_{len(layers_to_decompose[0])}_layers_decomposed.csv", index=False)