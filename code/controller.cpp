#include "controller.h"

template<typename Out>
static void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

static std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

inline
void check_solvers()
{
    int value= -1;
    read(c[READ], &value, sizeof(value));
    if(value != -1){
        u32 proc_num = (u32)value;
        close(p[READ]);
        close(p[WRITE]);
        close(c[READ]);
        close(c[WRITE]);
        LOG("Solver("<<value<<") with pid("<<map_proc_pid[proc_num]+") ended in "<<(_gettime() - map_proc_time[proc_num])<<"ns\n");
        if(kill(scheduler_pid, SIGINT) != 0)
        {
            perror("Failed to shut down scheduler\n");
        }
        else{
            LOG("Shut down signal sent correctly\n");
        }
    }
    return;
}


vector<u32> prune()
{
    arma::mat testset;
    arma::cube testX, predY;
    vector<u32> temp;
    ifstream fin;
    ofstream fout;
    string line;
    string content;
    if(csolvers <= 5)
    {
        return temp;
    }

    fin.open(trainFile);
    fout.open("load.csv");
    while(std::getline(fin, line))
    {
        line = line.substr(0,line.find("|"));
        fout << line << "\n";
    }
    fin.close();
    fout.close();

    data::Load("load.csv", testset, true);
    testX.set_size(21, testset.n_cols, 1);
    for (size_t i = 0; i < testset.n_cols - 1; i++)
    {
        testX.subcube(arma::span(), arma::span(i), arma::span()) = testset.submat(arma::span(), arma::span(i, i));
    }
    model.Predict(testX, predY);
    arma::mat checkY = predY.slice(predY.n_slices - 1);
    for (size_t i = 0; i < testset.n_cols - 1; i++)
    {
        double x = checkY(0,i);
        x = x + 0.5 - (x<0);
        int y = (int)x; 
        if(y)
        {
            LOG("Solver("<<i<<") with pid("<<map_proc_pid[(u32)i]<<") killed after "<<(_gettime() - map_proc_time[(u32)i])<<"ns\n");
            if(kill(scheduler_pid, SIGINT) != 0)
            {
                perror("Failed to shut down scheduler\n");
            }
            else
            {
                LOG("Shut down signal sent correctly\n");
            }
            csolvers -= 1;
            temp.push_back((u32)i);
        }

        if(csolvers <= 5)
        {
            break;
        }
    }
    return temp;
}

void launch_solver(u32 proc_num)
{
    u32 value;
    sigset_t blockMask, origMask;
    struct sigaction saIgnore, saOrigQuit, saOrigInt, saDefault;
    pid_t childPid;
    int status, savedErrno;
    string command;

    if(paramLoad)
    {
        string temp;
        getline(paramInput, temp);
        command = temp.substr(temp.find(":")+1,temp.length());
    }
    else
    {
        command = param->create_params();
        paramOutput << proc_num << ":" << command << "\n";
    }
    LOG("Solver("<<to_string(proc_num)<<") has params: "<<command<<"\n");
    
    pid_t pid = fork();
    if(pid == -1)
    {
        struct rlimit rlim;
        getrlimit(RLIMIT_NPROC, &rlim);
        LOG("Solver Forking Failed\n");
        LOG("RLIMIT_NPROC: Soft ("<<(u64)rlim.rlim_cur<<"), Hard("<<(u64)rlim.rlim_max<<"), Infinity("<<(u64)RLIM_INFINITY<<")\n");
    }
    else if(pid == 0)
    {
        LOG("Child("<<proc_num<<") solver\n");
        signal(SIGCHLD, SIG_DFL);

        sigemptyset(&blockMask);            /* Block SIGCHLD */
        sigaddset(&blockMask, SIGCHLD);
        sigprocmask(SIG_BLOCK, &blockMask, &origMask);

        // saIgnore.sa_handler = SIG_IGN;      /* Ignore SIGINT and SIGQUIT */
        // saIgnore.sa_flags = 0;
        // sigemptyset(&saIgnore.sa_mask);
        // sigaction(SIGINT, &saIgnore, &saOrigInt);
        // sigaction(SIGQUIT, &saIgnore, &saOrigQuit);

        switch (childPid = fork()) {
            case -1: {/* fork() failed */
                struct rlimit rlim;
                getrlimit(RLIMIT_NPROC, &rlim);
                LOG("Solver Forking Failed in Child Process\n");
                LOG("RLIMIT_NPROC: Soft ("<<(u64)rlim.rlim_cur<<"), Hard("<<(u64)rlim.rlim_max<<"), Infinity("<<(u64)RLIM_INFINITY<<")\n");
                status = -1;
                break;          /* Carry on to reset signal attributes */
            }

            case 0: {/* Child: exec command */

                /* We ignore possible error returns because the only specified error
                is for a failed exec(), and because errors in these calls can't
                affect the caller of system() (which is a separate process) */

                saDefault.sa_handler = SIG_DFL;
                saDefault.sa_flags = 0;
                sigemptyset(&saDefault.sa_mask);

                if (saOrigInt.sa_handler != SIG_IGN)
                    sigaction(SIGINT, &saDefault, NULL);
                if (saOrigQuit.sa_handler != SIG_IGN)
                    sigaction(SIGQUIT, &saDefault, NULL);

                sigprocmask(SIG_SETMASK, &origMask, NULL);

                write(p[WRITE], &proc_num, sizeof(proc_num));
                
                vector<string> cmds_vec = split(command, ' ');
                const char *exec_args[200];
                unsigned int i;
                exec_args[0] = filetype.c_str();
                exec_args[1] = ("-file:"+input).c_str();
                for(i = 2; i < cmds_vec.size(); i++)
                {
                    exec_args[i] = cmds_vec.at(i).c_str();
                }
                exec_args[i] = NULL;

                vector<string> env_vec;
                env_vec.push_back("PROCNUM="+proc_num);
                if(paramLoad)
                {
                    env_vec.push_back("FILENAME="+to_string(global_count)+"-"+trainFile);
                    env_vec.push_back("TRAIN=1");
                }
                else
                {
                    env_vec.push_back("FILENAME="+trainFile);
                    env_vec.push_back("TRAIN=0");
                }
                const char *env_args[200];
                for(i = 0; i < env_vec.size(); i++)
                {
                    env_args[i] = env_vec.at(i).c_str();
                }
                env_args[i] = NULL;
		
		execve("/home/sharjeel/z3/build/z3", (char *const*)exec_args, (char *const*)env_args); 
		//execve("/storage/home/hcoda1/3/skhan352/z3/build/z3", (char *const*)exec_args, (char *const*)env_args);

                _exit(127);                     /* We could not exec the shell */
            }
            default: /* Parent: wait for our child to terminate */
            {
                /* We must use waitpid() for this task; using wait() could inadvertently
                collect the status of one of the caller's other children */

                while (waitpid(childPid, &status, 0) == -1) {
                    if (errno != EINTR) {       /* Error other than EINTR */
                        status = -1;
                        break;                  /* So exit loop */
                    }
                }

                write(c[WRITE], &proc_num, sizeof(proc_num));

                break;
            }
        }
        close(p[READ]);
        close(p[WRITE]);
        close(c[READ]);
        close(c[WRITE]);
        _Exit(status);
    }   
    else 
    {
        //Add some lock
        read(p[READ], &value, sizeof(value));
        LOG("Received message from "<<value<<"\n");
        csolvers += 1;
        global_count += 1;
        map_proc_time[proc_num] = _gettime();
        map_proc_pid[proc_num] = pid;
    }
    
}


void relaunch_solvers(vector<u32> nsolvers)
{
    for(auto i : nsolvers)
    {
        launch_solver(i);
    }
}

void initial_solvers()
{
    u32 i;
    for(i = 0; i < nsolvers; ++i)
    {
        launch_solver(i);
    }
    LOG("Launched "<<nsolvers<<" solvers\n");
}

int main(int argc, char **argv)
{
    char *end;
    scheduler_pid = getpid();
    ofstream f;
    u32 epoch;    
    int option_index = 0;
    int opt;
    signal(SIGINT,  catch_control_c);
    signal(SIGSTOP, sigstop_handler);
    signal(SIGKILL, sigkill_handler);
    signal(SIGCHLD, SIG_IGN);

    while ((opt = getopt_long (argc, argv, "c:e:f:i:p:m:t:sh", long_options, &option_index)) != EOF)
    {
        switch(opt)
        {
            case 'c': 
                nsolvers = strtoul(optarg, &end, 10);
                break;
            case 'e': 
                epoch = strtoul(optarg, &end, 10);
                break;
            case 'f':
                filetype.append(optarg);
                break;
            case 'h':
                usage_and_exit(1);
                break;
            case 'i': 
                input.append(optarg);
                break;
            case 'l': 
                paramLoad = true;
                break;
            case 'm': 
                modelFile.append(optarg);
                break;
            case 'p': 
                paramFile.append(optarg);
                break;
            case 't': 
                trainFile.append(optarg);
                break;
            default: 
                perror("Error parsing Flags\n");
                LOG("Error parsing Flags\n");
                usage_and_exit(3);
        }
    }

    if(paramLoad)
    {
        paramInput.open(paramFile);
    }
    else
    {
        paramOutput.open(paramFile);
    }

    if(pipe(p) < 0){
        LOG("Pipe not working\n");
        exit(2);
    }

    if(pipe(c) < 0){
        LOG("Pipe not working\n");
        exit(2);
    }

    fcntl(c[READ], F_SETFL, fcntl(c[READ], F_GETFL) | O_NONBLOCK);

    LOG("Setup Params Class\n");
    param = new Params();

    data::Load(modelFile, "LSTMMulti", model);
    LOG("Setup Model from "<<modelFile<<"\n");

    f.open(trainFile);
    for(int i = 0; i < nsolvers; i++)
    {
        string temp(249, '_');
        f << temp << "\n";
    }
    f.close();
    LOG("Setup trainFile from "<<trainFile<<"\n");

    LOG("Controller PID("<<scheduler_pid<<") on "<<input<<" file\n");

    start_time = _gettime();
    initial_solvers();
    close(p[READ]);
    close(p[WRITE]);
    while(true)
    {
        sleep(epoch*1000);
        check_solvers();
        if(!paramLoad)
        {
            vector<u32> solvers = prune();
        }
        //relaunch_solvers(solvers);
    }
}
