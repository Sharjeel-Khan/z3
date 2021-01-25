# Z3 Multiprocessing 

## Files

z3_multiprocess_train.py: Main file that runs all processes in parallel to produce files for training the LSTM model. The flags are:

* **Testcase (-testcase) [Required]**: SMT or SAT input file 
* **Input (-input) [Required]**: Input file containing parameters for each process 
* **Output(-output) [Required]**: Output format for all the processes. For example, if input is test, the ith process will output to test-0.output.
* **Number of Processes (-num_procs) [Not Required, Default=1]**: Number of processes to use in Multiprocessing Z3 
* **All Processes Finish (-all_finish) [Not Required, Default=False]**: Allow all processes to finish normally. If not, it allows the first process to end then kills the rest.

z3_multiprocess.py (Edit this file Chris): Main file that runs all processes in parallel to LSTM that predicts whether the process is good to continue running or not. The flags are:
* **Testcase (-testcase) [Required]**: SMT or SAT input file 
* **Input (-input) [Required]**: Input file containing parameters for each process 
* **Model (-model) [Required]**: LSTM model file (.pt) is the lstm model used to predict whether process should continue the same or not
* **Number of Processes (-num_procs) [Not Required, Default=1]**: Number of processes to use in Multiprocessing Z3 
* **All Processes Finish (-all_finish) [Not Required, Default=False]**: Allow all processes to finish normally. If not, it allows the first process to end then kills the rest. Typically, you will only allow the first process to finish and kill the rest.

z3.pbs: Example file for Pace to run the process. To run it, just do qstat z3.pbs. Long processes take time so just run a bunch at the same time to not waste time because some examples take a lot of time.

combine.sh : Script that cleans up all the files and combines all processes's training data into one file. It also adds the 
* **Training Data File ($1) [Required]**: Training data filename for all the processes. For example, the parameter is test for files in the format
of test-i.output and output it to test.output.

addY.py: Script to add y-value for each dataset. The flags are:
* **Input File (-i) [Required]**: Input File 
* **Output File (-o) [Required]**: Output File

select.py: Script to do PCA analysis, normalize or just use SelectKBest. The flags are:
* **Input File (-i) [Required]**: Input File 
* **Output File (-o) [Required]**: Output File
* **Number of Features (-k) [Required]**: Number of features that you want to shrink the training dataset

training.py: Offline script to train the LSTM model and save it. I am planning to add checkpointing later on but not right now. The flags are:
* **Training Data (-input) [Required]**: Training data to train the LSTM model. 
* **Training Data (-input) [Required]**: Training data to train the LSTM model. 

lstm.py: Online script that loads the LSTM model and uses it to predict . (Not completed and might have bugs so you might have to look at this file.) The flags are:
* **Model (-model) [Required]**: LSTM model file (.pt) is the lstm model used to predict whether process
* **Test File (-file) [Required]**: File used to communicate with z3 process and use the output to make 


## System process steps

In my example, I will be talking about test file called test.cnf. You can generalize it by changing test.cnf to some other file whether it is cnf format or smt format. For the steps, you can either run them normally or put all the steps into a job file and run the job file.

First step, you should copy everything to the front path or your should change the job file (z3.pbs) for you to change the path.

Second step, you should run the training using multiprocessing on test.cnf. I am using input.txt as an example file.
```
python z3_multiprocessing_train.py -testcase test.cnf -input input.txt -output test -num_procs 10
```

Third step, you need combine all the files into one giant training data set. You need to step into the outputs folder (Create one before hand). This should remove all the output files and create one output file
```
cd outputs/
./combine.sh test
```

Fourth step, create the LSTM model from training set. In our case, it saved the model as test.pt.
```
cd ..
python training.py -test test_other.output -output test
```

Extra step, you can do an offline accuracy check for the model. 
```
cd ..
python accuracy.py -input test.output
```

Final step, you should run the main program that loads the LSTM model and predicts whether process should complete or not. (Didn't complete so it does not work. You will need to work on this meaning you will have to work on lstm.py file and z3_multiprocess.py)
```
python z3_multiprocess.py 
```

## Future Work to complete

These are the steps that we need to do in the future:

* We need to figure out how to make the LSTM model communicate with the z3 C++ process so the LSTM model can take the z3 process' stats and predict whether to kill it or change the parameters on the fly. 
    - Originally, I was just writing to a file and the LSTM process was reading the file and outputting to the main process.

* We need to change the timing within the z3 C++ process so we don't overwhelm the LSTM process with inputs coming too fast

* We need to figure out how to change solver attributes on the fly based on LSTM communication

* We need to figure out the best y values for LSTM to guess to allow best accuracy and effects for the Z3 process