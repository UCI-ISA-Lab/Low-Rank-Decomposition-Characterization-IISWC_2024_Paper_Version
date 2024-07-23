import os
import csv
import pandas

power_data = pandas.read_csv("power.log")

gpu0 = power_data[power_data["index"] == 0]
gpu1 = power_data[power_data["index"] == 1]
gpu2 = power_data[power_data["index"] == 2]
gpu3 = power_data[power_data["index"] == 3]

average_power_gpu0 = gpu0[" power.draw [W]"].mean()
average_power_gpu1 = gpu1[" power.draw [W]"].mean()
average_power_gpu2 = gpu2[" power.draw [W]"].mean()
average_power_gpu3 = gpu3[" power.draw [W]"].mean()

average_energy_gpu0 = gpu0[" power.draw [W]"].sum()
average_energy_gpu1 = gpu1[" power.draw [W]"].sum()
average_energy_gpu2 = gpu2[" power.draw [W]"].sum()
average_energy_gpu3 = gpu3[" power.draw [W]"].sum()

average_memory_usage_gpu0 = gpu0[" memory.used [MiB]"].mean()
average_memory_usage_gpu1 = gpu1[" memory.used [MiB]"].mean()
average_memory_usage_gpu2 = gpu2[" memory.used [MiB]"].mean()
average_memory_usage_gpu3 = gpu3[" memory.used [MiB]"].mean()

print(f"GPU0: avg power = {average_power_gpu0} \t energy = {average_energy_gpu0} \t memory usage = {average_memory_usage_gpu0}")
print(f"GPU1: avg power = {average_power_gpu1} \t energy = {average_energy_gpu1} \t memory usage = {average_memory_usage_gpu1}")
print(f"GPU2: avg power = {average_power_gpu2} \t energy = {average_energy_gpu2} \t memory usage = {average_memory_usage_gpu2}")
print(f"GPU3: avg power = {average_power_gpu3} \t energy = {average_energy_gpu3} \t memory usage = {average_memory_usage_gpu3}")