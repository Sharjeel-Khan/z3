#include "controller.h"

#include <sys/time.h>
#include <sys/resource.h>

inline
void check_finish_solvers()
{

    // for(i = 0; i < nsolvers; ++i)
    // {
    //     if(map_proc_pid.count(i) == 0){
    //         if(kill(scheduler_pid, SIGINT) != 0){
    //             perror("Failed to shut down scheduler\n");
    //         }else{
    //             LOG("Shut down signal sent correctly\n");
    //         }
    //     }
    // }
    return;
}


void prune()
{
    return;
}

void launch_solver(u32 proc_num)
{
    u32 value;
    sigset_t blockMask, origMask;
    struct sigaction saIgnore, saOrigQuit, saOrigInt, saDefault;
    pid_t childPid;
    int status, savedErrno;

    pid_t pid = fork();
    if(pid == -1)
    {
        struct rlimit rlim;
        getrlimit(RLIMIT_NPROC, &rlim);
        LOG("Solver Forking Failed\n");
        LOG("RLIMIT_NPROC: Soft ("<<(u64)rlim.rlim_cur<<"), Hard("<<(u64)rlim.rlim_max<<"), Infinity("<<(u64)RLIM_INFINITY<<")\n");
    }
    else if(pid == 0){
        LOG("Child("<<proc_num<<") solver\n");
        signal(SIGCHLD, SIG_DFL);

        sigemptyset(&blockMask);            /* Block SIGCHLD */
        sigaddset(&blockMask, SIGCHLD);
        sigprocmask(SIG_BLOCK, &blockMask, &origMask);

        saIgnore.sa_handler = SIG_IGN;      /* Ignore SIGINT and SIGQUIT */
        saIgnore.sa_flags = 0;
        sigemptyset(&saIgnore.sa_mask);
        sigaction(SIGINT, &saIgnore, &saOrigInt);
        sigaction(SIGQUIT, &saIgnore, &saOrigQuit);

        switch (childPid = fork()) {
            case -1: {/* fork() failed */
                struct rlimit rlim;
                getrlimit(RLIMIT_NPROC, &rlim);
                LOG("Solver Forking Failed in Child Process\n");
                LOG("RLIMIT_NPROC: Soft ("<<(u64)rlim.rlim_cur<<"), Hard("<<(u64)rlim.rlim_max<<"), Infinity("<<(u64)RLIM_INFINITY<<")\n");
                status = -1;
                break;          /* Carry on to reset signal attributes */
            }

            case 0: /* Child: exec command */

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

                // cporter: Original execl code looked like this:
                //execl("/bin/sh", "sh", "-c", command, (char *) NULL);
                // ... we're hard-coding taskset into the design for now:
                // TODO: get the taskset path from the command.
                write(p[WRITE], &proc_num, sizeof(proc_num));

                execv("/usr/bin/taskset", (char *const*)command);

                _exit(127);                     /* We could not exec the shell */

            default: /* Parent: wait for our child to terminate */

                /* We must use waitpid() for this task; using wait() could inadvertently
                collect the status of one of the caller's other children */

                while (waitpid(childPid, &status, 0) == -1) {
                    if (errno != EINTR) {       /* Error other than EINTR */
                        status = -1;
                        break;                  /* So exit loop */
                    }
                }

                bes_proc_exit(childPid, proc_num);

                break;




        close(p[READ]);
        close(p[WRITE]);
        _Exit(status);
    }   
    else 
    {
        //Add some lock
        read(p[READ], &value, sizeof(value));
        LOG("Received message from "<<value<<"\n");
        num_solvers += 1;
        map_proc_time[proc_num] = _gettime();
        map_proc_pid[proc_num] = pid;
    }
    
}


void relaunch_solvers(u64 nsolvers)
{
    u32 i;
    for(i = 0; i < nsolvers; ++i)
    {
        if(map_proc_pid.count(i) == 0){
            launch_solver(i);
        }
    }
}

void initial_solvers(u64 nsolvers)
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
    
    signal(SIGINT,  catch_control_c);
    signal(SIGSTOP, sigstop_handler);
    signal(SIGKILL, sigkill_handler);
    signal(SIGCHLD, SIG_IGN);

    if(argc != 3)
    {
        usage_and_exit(1);
    }

    u64 nsolvers = strtoull(argv[1], &end, 10);

    if(pipe(p) < 0){
        LOG("Pipe not working\n");
        usage_and_exit(2);
    }

    LOG("Setup Params Class\n");
    params = make_unique<Params>();

    LOG("Setup Model from "<<argv[2]<<"\n");

    LOG("Controller PID("<<scheduler_pid<<") with "<<nsolvers<<" solvers\n");

    start_time = _gettime();
    initial_solvers(nsolvers);
    close(p[READ]);
    close(p[WRITE]);
    // while(true)
    // {
    //     sleep(1000);
    //     check_finish_solvers();
    //     prune();
    //     relaunch_solvers(nsolvers);
    // }
}
