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

def clamp_number(number, total_digits):
    digits_before_decimal = len(str(int(number)))
    digits_after_decimal = total_digits - digits_before_decimal
    factor = 10 ** digits_after_decimal
    return round(number * factor) / factor

def generate_dataset(fr_range, n_range, fr_interval, n_interval, output_path, save_to_file=False):
    fr_values = np.arange(fr_range[0], fr_range[1] + fr_interval, fr_interval)
    n_values = np.arange(n_range[0], n_range[1] + n_interval, n_interval)

    data = []
    for fr in fr_values:
        for n in n_values:
            lifetime = generate_lifetime(fr, n)
            data.append([fr, n, clamp_number(lifetime, 10)])
    
    df = pd.DataFrame(data, columns=['Fr', 'n', 'Lifetime'])
    if save_to_file:
        df.to_csv(output_path, index=False)
        print(f'Dataset generated and saved to {output_path}')
    return df

# generate_dataset(fr_range, n_range, fr_interval, n_interval, output_path, save_to_file=True)
