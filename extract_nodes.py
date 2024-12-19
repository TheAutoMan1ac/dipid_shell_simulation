import csv
import re
import numpy as np
from pathlib import Path
import pandas as pd
from simulation_assembly import MolecularDynamicsSimulation

output_file = "node_degrees_sorted.csv"
input_folder = Path("simulations")

# Define the maximum degree we explicitly count
max_degree = 11  # We count from 0 to 11 explicitly

# Regular expressions to extract values from filenames
dt_pattern = re.compile(r"_dt([0-9.]+)")
a_pattern = re.compile(r"_a([0-9.]+)")
bsize_pattern = re.compile(r"_Bsize([0-9.]+)")

# Data collection list
data = []

for filepath in input_folder.glob("*.pkl"):
    # Extract parameters from filename
    filename = filepath.name
    dt_match = dt_pattern.search(filename)
    a_match = a_pattern.search(filename)
    bsize_match = bsize_pattern.search(filename)

    if not (dt_match and a_match and bsize_match):
        print(f"Skipping file {filename}: required parameters not found.")
        continue

    dt_value = float(dt_match.group(1))
    a_value = float(a_match.group(1))
    bsize_value = int(bsize_match.group(1))

    # Load the simulation
    sim_instance = MolecularDynamicsSimulation.load_state(filepath)
    last_state = sim_instance.state_trajectory[-1]
    topology = last_state['topology']

    # Initialize degree counts
    degree_array = np.zeros(max_degree + 2, dtype=int)  # 0..11 plus one for >=12

    for node in topology.nodes():
        degree = topology.degree(node)
        if degree <= max_degree:
            degree_array[degree] += 1
        else:
            degree_array[-1] += 1  # count for >=12

    # Append data for this file
    row = [filename, dt_value, a_value, bsize_value] + degree_array.tolist()
    data.append(row)

# Create a DataFrame
columns = ["filename", "dt", "a", "Bsize"] + [f"degree_{d}" for d in range(max_degree + 1)] + ["degree_12_plus"]
df = pd.DataFrame(data, columns=columns)

# Sort the DataFrame
sorted_df = df.sort_values(by=["dt", "a", "Bsize"], ascending=[True, True, True])

# Save to CSV
sorted_df.to_csv(output_file, index=False)