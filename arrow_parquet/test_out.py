#!/usr/bin/env python3
import pandas as pd


if __name__ == '__main__':
    data = {
        "calories": [420.5, 380.0, 390.6],
        "duration": [50, 40, 45]
    }
    df = pd.DataFrame(data)
    df.to_parquet('py_out.parquet')
