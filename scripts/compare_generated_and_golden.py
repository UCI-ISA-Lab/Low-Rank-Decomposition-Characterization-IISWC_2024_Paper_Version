import pandas as pd
import numpy as np

benchmarks = "arc_easy,arc_challenge,hellaswag,mmlu,truthfulqa,winogrande"
benchmark_names = "ARC Easy,ARC Challenge,HellaSwag,MMLU,TruthfulQA,WinoGrande"

paper_accuracy_file = "../paper_data/accuracy_impact_data.csv"
paper_inference_time_file = "../paper_data/inference_time.csv"
paper_power_memory_file = "../paper_data/power_memory.csv"

generated_accuracy_file = "../decomposed-lm-evaluation-harness/output_dir/accuracy_impact_data.csv"
generated_inference_time_file = "../decomposed-lm-evaluation-harness/output_dir/inference_time.csv"
generated_power_memory_file = "../decomposed-lm-evaluation-harness/output_dir/power_memory.csv"

paper_accuracy_df = pd.read_csv(paper_accuracy_file)
paper_inference_time_df = pd.read_csv(paper_inference_time_file)
paper_power_memory_df = pd.read_csv(paper_power_memory_file)

generated_accuracy_df = pd.read_csv(generated_accuracy_file)
generated_inference_time_df = pd.read_csv(generated_inference_time_file)
generated_power_memory_df = pd.read_csv(generated_power_memory_file)

pattern = "-----------------------------"

#Accuracy Impact Data Validation
print(f"\n\n\n{pattern}\nAccuracy Data Validation\n{pattern}")
mean_sq_error_df_list = []
for benchmark, benchmark_name in zip(benchmarks.split(","), benchmark_names.split(",")):
    paper_data = paper_accuracy_df[benchmark_name]
    generated_data = generated_accuracy_df[benchmark_name]
    mean_sq_error = np.linalg.norm(paper_data - generated_data)
    df = pd.DataFrame({"Benchmark": benchmark_name, "Mean Square Error": mean_sq_error}, index=[0])
    mean_sq_error_df_list.append(df)
    # print(f"Benchmark: {benchmark_name}\tMean Square Error: {mean_sq_error}")
mean_sq_error_df = pd.concat(mean_sq_error_df_list, ignore_index=True)
print(mean_sq_error_df)

#Inference Time Data Validation
print(f"\n\n\n{pattern}\nInference Time Data Validation\n{pattern}")
mean_sq_error_df_list = []
for benchmark, benchmark_name in zip(benchmarks.split(","), benchmark_names.split(",")):
    paper_data = paper_inference_time_df.loc[paper_inference_time_df["Benchmark"] == benchmark]["Inference Time Per Batch"]
    generated_data = generated_inference_time_df.loc[paper_inference_time_df["Benchmark"] == benchmark]["Inference Time Per Batch"]
    mean_sq_error = np.linalg.norm(paper_data - generated_data)
    df = pd.DataFrame({"Benchmark": benchmark_name, "Mean Square Error": mean_sq_error}, index=[0])
    mean_sq_error_df_list.append(df)
    # print(f"Benchmark: {benchmark_name}\tMean Square Error: {mean_sq_error}")
mean_sq_error_df = pd.concat(mean_sq_error_df_list, ignore_index=True)
print(mean_sq_error_df)

#Energy Data Validation
print(f"\n\n\n{pattern}\nEnergy Data Validation\n{pattern}")
mean_sq_error_df_list = []
for benchmark, benchmark_name in zip(benchmarks.split(","), benchmark_names.split(",")):
    paper_data = paper_power_memory_df.loc[paper_inference_time_df["Benchmark"] == benchmark]["Average Energy Per Batch"]
    generated_data = generated_power_memory_df.loc[paper_inference_time_df["Benchmark"] == benchmark]["Average Energy Per Batch"]
    mean_sq_error = np.linalg.norm(paper_data - generated_data)
    df = pd.DataFrame({"Benchmark": benchmark_name, "Mean Square Error": mean_sq_error}, index=[0])
    mean_sq_error_df_list.append(df)
    # print(f"Benchmark: {benchmark_name}\tMean Square Error: {mean_sq_error}")
mean_sq_error_df = pd.concat(mean_sq_error_df_list, ignore_index=True)
print(mean_sq_error_df)

#Memory Usage Data Validation
print(f"\n\n\n{pattern}\nMemory Usage Data Validation\n{pattern}")
mean_sq_error_df_list = []
for benchmark, benchmark_name in zip(benchmarks.split(","), benchmark_names.split(",")):
    paper_data = paper_power_memory_df.loc[paper_inference_time_df["Benchmark"] == benchmark]["Average Memory Usage"]
    generated_data = generated_power_memory_df.loc[paper_inference_time_df["Benchmark"] == benchmark]["Average Memory Usage"]
    mean_sq_error = np.linalg.norm(paper_data - generated_data)
    df = pd.DataFrame({"Benchmark": benchmark_name, "Mean Square Error": mean_sq_error}, index=[0])
    mean_sq_error_df_list.append(df)
    # print(f"Benchmark: {benchmark_name}\tMean Square Error: {mean_sq_error}")
mean_sq_error_df = pd.concat(mean_sq_error_df_list, ignore_index=True)
print(mean_sq_error_df)