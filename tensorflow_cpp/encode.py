#!/usr/bin/env python3
import pandas as pd
import numpy as np
import tensorflow as tf


def load_data(filename):
    df = pd.read_csv(filename, compression='gzip')
    if 'weight' not in df.columns:
        df['weight'] = 1.0
    return df


def load_model(model_dirname):
    model = tf.keras.models.load_model(model_dirname)
    return model


def main():
    CVs = ['X', 'Y']
    # load data
    data = load_data('dx0.1_dy1.0.csv.gz')
    # load model
    model = load_model('best_encoder_model/')
    # encode data
    x = tf.Variable(tf.convert_to_tensor(data[CVs].to_numpy()))
    with tf.GradientTape() as tape:
        tape.watch(x)
        nn_output_data = model(x)
    grad = tape.gradient(nn_output_data, x)
    print(grad)
    # save encoded data
    print(nn_output_data)
    nn_output_data_df = pd.DataFrame(nn_output_data)
    nn_output_data_df.columns = ['committor']
    nn_output_data_df['grad_x'] = grad.numpy()[:, 0]
    nn_output_data_df['grad_y'] = grad.numpy()[:, 1]
    nn_output_data_df.to_csv('encoded_py.csv', index=False)


if __name__ == '__main__':
    main()
