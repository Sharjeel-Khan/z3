import os
from multiprocessing import Process
from time import time
import argparse

# List of processes
process_list = []

# LSTM processes
lstm_list = []

# Time for each process
ptime = []

# Parameters list
parameters = []

# Boolean type for all finish flag
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Execute Z3 process with parameters and save it into output file. 
def z3_exe(testcase, parameters, output):
    output = output + ".output"
    os.system("./z3_exe " + parameters + " " + testcase + " > " + output)

# Execute python script reading lstm model
def lstm_exe(model, output):
    output = output + ".output"
    os.system("python lstm.py -model "+model+" -file "+output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiprocessing Z3')
    parser.add_argument('-testcase', dest='testcase', type=str, required=True,  help='Testcase to run')
    parser.add_argument('-input', dest='input', type=str, required=True,  help='Input file with different parameters')
    parser.add_argument('-model', dest='model', type=str, required=True,  help='LSTM Model file')
    parser.add_argument('-output', dest='output', type=str, required=True,  help='Output file Name')
    parser.add_argument('-num_procs', dest='num_procs', type=int, required=True,  help='Number of processes')
    parser.add_argument('-all_finish', dest='all_finish', type=str2bool, required=False, default=False,  help='Allow processes to finish or not')

    args = parser.parse_args()
    N = args.num_procs

    print("Program Execution: ")
    print("Testcase: "+args.testcase)
    print("Input File: "+args.input)
    print("Model: "+args.model)
    print("Num of Processes: "+str(N))
    print("Allow all processes to finish: "+str(args.all_finish))

    # Read parameters from the input file
    with open(args.input, 'r') as file:
        line = file.readline().rstrip()
        while(line):
            parameters.append(line)
            line = file.readline().rstrip()
        
        while(len(parameters) != N):
            parameters.append("")
    
    # Create each process and start executing it alongside a lstm process for this process
    # If we have 10 processes, we will have 10 LSTM processes where each process will communicate with its
    # own LSTM model.
    for i in range(N):
        p = Process(target=z3_exe, args=(args.testcase, parameters[i], args.output+"-"+str(i)))
        l = Process(target=lstm_exe, args=(args.model, args.output+"-"+str(i)))
        p.start()
        l.start()
        process_list.append(p)
        lstm_list.append(l)
        ptime.append(time())

    # Wait until all processes finish
    if(args.all_finish):
        for i in range(N):
            process_list[i].join()
            print("Time for "+str(i)+" iteration: "+str(time() - ptime[i]))
    else:
        # Wait until first process to end and terminate other processes 
        done = False
        while(not done):
            for i in range(N):
                if(not process_list[i].is_alive()):
                    print("Time for "+str(i)+" iteration: "+str(time() - ptime[i]))
                    done = True
                    
        for i in range(N):
            if(process_list[i].is_alive()):
                process_list[i].terminate()


