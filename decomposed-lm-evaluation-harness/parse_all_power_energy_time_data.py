import os
import csv
import pandas as pd
import json

output_dir = "power_output"
inference_time_dir = "jsonl_output"
power_mem_dir = "power_logs"

benchmarks = "arc_easy,arc_challenge,hellaswag,mmlu,truthfulqa,winogrande"
print_name_benchmarks = "ARC Easy,ARC Challenge,HellaSwag,MMLU,TruthfulQA,WinoGrande"

#Inference Time
files = os.listdir(os.path.join(output_dir, inference_time_dir))
df_list = []
for benchmark in benchmarks.split(","):
    results = {}
    for file in files:
        if(file.startswith(benchmark)):
            with open(os.path.join(output_dir, inference_time_dir, file), "r") as input_file:
                data = json.load(input_file)
            results["Layers Decomposed"] = [data["layers_decomposed"]]
            results["Matrices Decomposed"] = [data["matrices_decomposed"]]
            results["Rank"] = data["decomposition_rank"]
            results["Batch Size"] = data["config"]["batch_sizes"][0]
            results["Dataset Size"] = data["dataset_size"]
            results["Total Inference Time(s)"] = data["inference_time"]
            results["Benchmark"] = benchmark
            results["Parameter Reduction(%)"] = data["decomposition_percentage"]
            num_batches = results["Dataset Size"] / results["Batch Size"]
            inference_time_per_batch = results["Total Inference Time(s)"] / num_batches
            results["Inference Time Per Batch"] = inference_time_per_batch
            df = pd.DataFrame(results, index=[0])
            df_list.append(df)
final_df = pd.concat(df_list, ignore_index=True)
# print(final_df)
final_df.to_csv("output_dir/inference_time.csv")

#Power and Memory Usage/Utilization
power_files = os.listdir(os.path.join(output_dir, power_mem_dir))
json_files = os.listdir(os.path.join(output_dir, inference_time_dir))
json_files = sorted(json_files)
power_files = sorted(power_files)

df_list = []
for benchmark in benchmarks.split(","):
    results = {}
    for power_file, json_file in zip(power_files, json_files):
        if(json_file.startswith(benchmark)):
            with open(os.path.join(output_dir, inference_time_dir, json_file), "r") as input_file:
                time_data = json.load(input_file)
            results["Layers Decomposed"] = [time_data["layers_decomposed"]]
            results["Matrices Decomposed"] = [time_data["matrices_decomposed"]]
            results["Rank"] = time_data["decomposition_rank"]
            results["Batch Size"] = time_data["config"]["batch_sizes"][0]
            results["Dataset Size"] = time_data["dataset_size"]
            num_batches = results["Dataset Size"] / results["Batch Size"]
            results["Number of Batches"] = num_batches
            results["Total Inference Time(s)"] = time_data["inference_time"]
            results["Benchmark"] = benchmark
            results["Parameter Reduction(%)"] = time_data["decomposition_percentage"]
            inference_time_per_batch = results["Total Inference Time(s)"] / num_batches
            results["Inference Time Per Batch"] = inference_time_per_batch
            with open(os.path.join(output_dir, power_mem_dir, power_file)) as power_log_file:
                power_data = pd.read_csv(power_log_file, skipinitialspace=True)
            power_data["power.draw [W]"] = power_data["power.draw [W]"].str.replace(" W","")
            power_data["memory.used [MiB]"] = power_data["memory.used [MiB]"].str.replace(" MiB","")
            power_data["utilization.memory [%]"] = power_data["utilization.memory [%]"].str.replace(" %","")
            power_data["utilization.gpu [%]"] = power_data["utilization.gpu [%]"].str.replace(" %","")
            power_data["power.draw [W]"]         = power_data["power.draw [W]"].astype(float)
            power_data["memory.used [MiB]"]      = power_data["memory.used [MiB]"].astype(int)
            power_data["utilization.memory [%]"] = power_data["utilization.memory [%]"].astype(int)
            power_data["utilization.gpu [%]"]    = power_data["utilization.gpu [%]"].astype(int)
            power_data = power_data[power_data["utilization.gpu [%]"] != 0]
            gpu0 = power_data[power_data["index"] == 0]
            gpu1 = power_data[power_data["index"] == 1]
            gpu2 = power_data[power_data["index"] == 2]
            gpu3 = power_data[power_data["index"] == 3]
            
            average_power_gpu0 = gpu0["power.draw [W]"].mean()
            average_power_gpu1 = gpu1["power.draw [W]"].mean()
            average_power_gpu2 = gpu2["power.draw [W]"].mean()
            average_power_gpu3 = gpu3["power.draw [W]"].mean()
            total_average_power = average_power_gpu0 + average_power_gpu1 + average_power_gpu2 + average_power_gpu3
            
            average_energy_gpu0 = gpu0["power.draw [W]"].sum()
            average_energy_gpu1 = gpu1["power.draw [W]"].sum()
            average_energy_gpu2 = gpu2["power.draw [W]"].sum()
            average_energy_gpu3 = gpu3["power.draw [W]"].sum()
            total_average_energy = average_energy_gpu0 + average_energy_gpu1 + average_energy_gpu2 + average_energy_gpu3
            # total_average_energy = total_average_power * results["Total Inference Time(s)"]
            
            
            num_batches_per_gpu = num_batches / 4
            average_energy_per_batch_gpu0 = average_energy_gpu0 / num_batches_per_gpu
            average_energy_per_batch_gpu1 = average_energy_gpu1 / num_batches_per_gpu
            average_energy_per_batch_gpu2 = average_energy_gpu2 / num_batches_per_gpu
            average_energy_per_batch_gpu3 = average_energy_gpu3 / num_batches_per_gpu
            # average_energy_per_batch = (average_energy_per_batch_gpu0 + average_energy_per_batch_gpu1 + average_energy_per_batch_gpu2 + average_energy_per_batch_gpu3) / 4
            average_energy_per_batch = total_average_energy / num_batches
            
            average_memory_usage_gpu0 = gpu0["memory.used [MiB]"].mean()
            average_memory_usage_gpu1 = gpu1["memory.used [MiB]"].mean()
            average_memory_usage_gpu2 = gpu2["memory.used [MiB]"].mean()
            average_memory_usage_gpu3 = gpu3["memory.used [MiB]"].mean()
            average_memory_usage = (average_memory_usage_gpu0 + average_memory_usage_gpu1 + average_memory_usage_gpu2 + average_memory_usage_gpu3) / 4
            
            average_memory_utlization_gpu0 = gpu0["utilization.gpu [%]"].mean()
            average_memory_utlization_gpu1 = gpu1["utilization.gpu [%]"].mean()
            average_memory_utlization_gpu2 = gpu2["utilization.gpu [%]"].mean()
            average_memory_utlization_gpu3 = gpu3["utilization.gpu [%]"].mean()
            average_memory_utlization = (average_memory_utlization_gpu0 + average_memory_utlization_gpu1 + average_memory_utlization_gpu2 + average_memory_utlization_gpu3) / 4
            
            results["Total Average Power"] = total_average_power
            results["Total Average Energy"] = total_average_energy
            results["Average Energy Per Batch"] = average_energy_per_batch
            results["Average Memory Usage"] = average_memory_usage
            results["Average Memory Utlization"] = average_memory_utlization
            df = pd.DataFrame(results, index=[0])
            df_list.append(df)        
final_df = pd.concat(df_list, ignore_index=True)
# print(final_df)
final_df.to_csv("output_dir/power_memory.csv")