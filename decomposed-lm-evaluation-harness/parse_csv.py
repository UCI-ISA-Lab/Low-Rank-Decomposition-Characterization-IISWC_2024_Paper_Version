import os
import csv
import pandas as pd

base_dir = "output_dir"
files = os.listdir(base_dir)
files = [file for file in files if file.endswith("csv")]

benchmarks = "ARC Easy,ARC Challenge,HellaSwag,MMLU,TruthfulQA,WinoGrande,GSM8K"

for file in files[1:]:
    with open(os.path.join(base_dir, file), "r") as input_file:
        df = pd.read_csv(input_file)
        for benchmark in benchmarks.split(","):
            try:
                df[benchmark] = df[benchmark].apply(lambda x: x * 100)
            except:
                pass
        df.to_csv(os.path.join(base_dir, "processed_sweep_data", file))