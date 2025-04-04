import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, preprocessing
import argparse

def neighborhood_regression(mat):
    # iterate through rows of mat
    betas = []
    for row_idx, row in enumerate(mat):
        X = np.delete(mat.copy(), row_idx, axis=0)
        reg = linear_model.Lasso(alpha=0.1, max_iter=10000)
        reg.fit(X.T, row)
        betas.append(reg.coef_)
 
    betas = np.asarray(add_diagonal(betas))
    return betas

def add_diagonal(mat, value=1.0):
    new_mat = []
    for row_idx, row in enumerate(mat):
        row = list(row)
        new_row = row[:row_idx] + [value] + row[row_idx:]
        new_mat.append(new_row)
    new_mat = np.asarray(new_mat)
    return new_mat


def main(args):
    df = pd.read_pickle(args.in_path)
    print(df.head())

    row_labels = df['Intervention'].tolist()
    intervention_data = df.loc[:, df.columns != 'Intervention'].to_numpy()

    # preprocess data
    scaler = preprocessing.StandardScaler()
    intervention_data = scaler.fit_transform(intervention_data)

    # lasso regression
    betas = neighborhood_regression(intervention_data)

    # make output directory
    plt.figure(figsize=(6,10))
    im = plt.imshow(intervention_data, interpolation='nearest', cmap='RdBu_r')
    plt.colorbar(im)
    plt.xticks(())
    plt.yticks(np.arange(len(row_labels)), labels=row_labels)
    plt.title("Data")
    plt.show()
    plt.close()

    assert len(row_labels) == betas.shape[0]

    # plot covariance
    plt.figure(figsize=(15, 15))
    im = plt.imshow(betas, interpolation='nearest', cmap='RdBu_r')
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.clim(-1.0, 1.0)
    plt.xticks(np.arange(len(row_labels)), labels=row_labels, rotation='vertical')
    plt.yticks(np.arange(len(row_labels)), labels=row_labels)
    plt.title("Betas")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--in_path', type=str, default=None) # path to WID directory
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    main(args)