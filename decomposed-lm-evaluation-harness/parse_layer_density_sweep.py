import os
import json
import pandas as pd

output_dir = "output_dir"
files = [file for file in os.listdir(output_dir) if (file.startswith("layer_density_5L") and file.endswith("jsonl"))]
benchmarks = "arc_easy,arc_challenge,hellaswag,mmlu,truthfulqa,winogrande"
print_name_benchmarks = "ARC Easy,ARC Challenge,HellaSwag,MMLU,TruthfulQA,WinoGrande"
df_list = []
for file in files:
    with open(os.path.join(output_dir, file), "r") as input_file:
        data = json.load(input_file)
    results = {
        "Layers Decomposed": [data["layers_decomposed"]],
        "Matrices Decomposed": [data["matrices_decomposed"]],
        "Rank": data["decomposition_rank"],
        "Decomposition Percentage": data["decomposition_percentage"],
    }
    for benchmark, benchmark_name in zip(benchmarks.split(","), print_name_benchmarks.split(",")):
        results[benchmark_name] = data["results"][benchmark]["acc,none"] * 100
    df = pd.DataFrame(results, index=[0])
    df_list.append(df)
res_df = pd.concat(df_list, ignore_index=True)

res_df.to_csv("output_dir/processed_sweep_data/layer_density_sweep.csv", index=False)