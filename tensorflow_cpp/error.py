#!/usr/bin/env python3
import pandas as pd
import numpy as np


def main():
    output_cpp = pd.read_csv('encoded_cpp.csv').to_numpy()
    output_py = pd.read_csv('encoded_py.csv').to_numpy()
    error = np.sqrt(np.mean(np.square(output_cpp - output_py)))
    print(f'Error = {error}')


if __name__ == '__main__':
    main()
