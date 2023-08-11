import numpy as np
import pandas as pd
from config import config

# Modify the ranges, intervals, and output path as needed
fr_range = (200, 4000)
n_range = (100, 3500)
fr_interval = 100
n_interval = 100
output_path = config["dataset_path"]

def generate_lifetime(fr, n):
    return 4.1378625767 * 10**17 * fr ** (-10/3) * n ** (-1.0)

def generate_dataset(fr_range, n_range, fr_interval, n_interval, output_path):
    fr_values = np.arange(fr_range[0], fr_range[1], fr_interval)
    n_values = np.arange(n_range[0], n_range[1], n_interval)

    # Append the end value if it's not included and it's a valid step in the sequence
    if (fr_range[1] - fr_range[0]) % fr_interval == 0 and fr_values[-1] != fr_range[1]:
        fr_values = np.append(fr_values, fr_range[1])
    if (n_range[1] - n_range[0]) % n_interval == 0 and n_values[-1] != n_range[1]:
        n_values = np.append(n_values, n_range[1])

    data = []
    for fr in fr_values:
        for n in n_values:
            lifetime = generate_lifetime(fr, n)
            data.append([fr, n, format(lifetime, ".10")])
    
    # Save data to a CSV file
    df = pd.DataFrame(data, columns=['Fr', 'n', 'Lifetime'])
    df.to_csv(output_path, index=False)
    print(f'Dataset generated and saved to {output_path}')

generate_dataset(fr_range, n_range, fr_interval, n_interval, output_path)
