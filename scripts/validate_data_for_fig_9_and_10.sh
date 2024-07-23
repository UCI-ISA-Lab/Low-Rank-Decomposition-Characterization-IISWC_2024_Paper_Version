# Change Working directory to lm-evaluation harness
cd ../decomposed-lm-evaluation-harness

# Run all the benchmarks on the Llama-2-7B model and generate accuracy, inference time, energy and power usage data
python power_run.py

# Parse the generated data
python parse_all_power_energy_time_data.py

cd ../scripts
# Compare the generated data with the data used in the paper (ground truth)
python compare_generated_and_golden.py