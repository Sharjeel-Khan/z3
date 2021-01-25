import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiprocessing Z3')
    parser.add_argument('-i', dest='input', type=str, required=True,  help='Input File')
    parser.add_argument('-o', dest='output', type=str, required=True,  help='Output File')
    args = parser.parse_args()

    input_file = np.loadtxt(args.input, delimiter=",")

    #Initialize y values
    zeros = np.zeros((input_file.shape[0],1))

    N = input_file.shape[0]
    
    # Add it to the last column
    input_file = np.concatenate((input_file, zeros), axis=1)

    # For now we are checking if it will finish in the next 10 steps so just make the last 
    # 10 datapoints supposedly 1 (true)
    input_file[N-10:N,-1] = 1

    np.savetxt(args.output, input_file, delimiter=",")

    