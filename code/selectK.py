import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiprocessing Z3')
    parser.add_argument('-i', dest='input', type=str, required=True,  help='Input File')
    parser.add_argument('-o', dest='output', type=str, required=True,  help='Output File')
    parser.add_argument('-k', dest='k', type=int, required=True,  help='Selection Value')
    args = parser.parse_args()

    input_file = np.loadtxt(args.input, delimiter=",")
    D = input_file.shape[1]
    
    # Create X and y values from dataset
    X = input_file[:,0:D-1]
    y = input_file[:,D-1]

    # PCA Anaylsis (I just commented out because there was a better way keeping the same attributes)
    #pca = PCA(n_components=args.k)
    #X_new = pca.fit_transform(X, y)

    selectBest = SelectKBest(chi2, k=args.k)
    X_new = selectBest.fit_transform(X, y)
    y = y.reshape(y.shape[0], 1)
    input_file = np.concatenate((X_new,y), axis=1)
    #print(selectBest.pvalues_)

    np.savetxt(args.output, input_file, delimiter=",")

    