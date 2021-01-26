#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <map>
#include <set>
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <assert.h>
#include <string>
#include <iterator>
#include <sched.h>
#include <random>

using namespace std;

#ifdef DEBUG
static ofstream logFile("controller.log", ios::out);
#define LOG(str) do { logFile << _gettime() << " TRACE: " << __FUNCTION__ << " " << str; cout.flush(); } while( false )
#define LOG_NO_TIME(str) do { cout << str; cout.flush(); } while( false )
#else
#define LOG(str) do { } while ( false )
#define LOG_NO_TIME(str) do { } while ( false )
#endif

#define READ 0
#define WRITE 1

/****************************************************************
 * Types
 ****************************************************************/
typedef unsigned long long int u64;
typedef unsigned long int u32;

/****************************************************************
 * Global Variables
 ****************************************************************/
pid_t scheduler_pid;
u64 start_time;
u32 num_solvers = 0;
map<u32, u64> map_proc_time;
map<u32, pid_t> map_proc_pid;
unique_ptr<Params> params;
string filetype;
string input;
int p[2];
int c[2];


/****************************************************************
 * Timing and Usage Functions
 ****************************************************************/

static inline
u64 _gettime(void)
{
    struct timespec tv = {0};
    if(clock_gettime(CLOCK_REALTIME, &tv) != 0){
        return 0;
    }
    return (((long long) tv.tv_sec * 1000000000L ) + (long long) (tv.tv_nsec));
}

inline
void usage_and_exit(int rc)
{
    LOG("Usage: ./controller <num-cores> <model-file> <filetype> <input>\n");
    exit(rc);
}

/****************************************************************
 * Signal Handlers
 ****************************************************************/

inline
void catch_control_c(int unused)
{
    LOG("Closing Controller\n");
    u64 end_time = _gettime() - start_time;
    LOG("Total_Time: "<<(double)end_time * 1e-9<<"(s).\n");
    fflush(stdout);
    fflush(stderr);
    exit(0);
}

inline
void sigstop_handler(int sig)
{
    LOG("Inside sigstop handler\n");
}

inline
void sigkill_handler(int sig)
{
    LOG("Inside sigkill handler\n");
}

inline
void sigsegv_handler(int sig)
{
    LOG("Inside sigsegv handler\n");
}

/****************************************************************
 * Torch Model
 ****************************************************************/


/****************************************************************
 * Params Properties
 ****************************************************************/

class Params
{
    vector<string> phase_string;
    vector<string> restart_string;
    vector<string> branching_heuristic_string;
    vector<string> bool_string;

    /****************************************************************
     * Params Randomize Generators
     ****************************************************************/
    default_random_engine generator;
    uniform_int_distribution<u32> phase_distribution;
    uniform_int_distribution<u32> search_conflicts_distribution;
    uniform_int_distribution<u32> rephase_base_distribution;
    uniform_int_distribution<u32> reorder_base_distribution;
    uniform_int_distribution<u32> reorder_activity_scale_distribution;
    uniform_int_distribution<u32> restart_distribution;
    uniform_int_distribution<u32> restart_initial_distribution;
    uniform_real_distribution<double> restart_factor_distribution;
    uniform_real_distribution<double> restart_margin_distribution;
    uniform_real_distribution<double> restart_emafastglue_distribution;
    uniform_real_distribution<double> restart_emaslowglue_distribution;
    uniform_int_distribution<u32> variable_decay_distribution;
    uniform_int_distribution<u32> branching_heuristic_distribution;
    uniform_int_distribution<u32> branching_anti_exploration_distribution;
    uniform_real_distribution<double> random_freq_distribution;
    uniform_int_distribution<u32> seed_distribution;
    uniform_int_distribution<u32> burst_search_distribution;
    uniform_int_distribution<u32> enable_pre_simplify_distribution;
    uniform_int_distribution<u32> simplify_delay_distribution;
    uniform_int_distribution<u32> backtrack_scopes_distribution;
    uniform_int_distribution<u32> backtrack_conflicts_distribution;


public:
    Params() : phase_distribution(0,4), search_conflicts_distribution(400,UINT_MAX), rephase_base_distribution(1000,UINT_MAX), \
    reorder_base_distribution(1000,UINT_MAX), reorder_activity_scale_distribution(100,UINT_MAX), restart_distribution(0,1), \
    restart_initial_distribution(0,100), restart_factor_distribution(1.5,10.0), restart_margin_distribution(1.1,10.0), \
    restart_emafastglue_distribution(0.03,1.0), restart_emaslowglue_distribution(0.00001,1.0), variable_decay_distribution(100,UINT_MAX), \
    branching_heuristic_distribution(0,1), branching_anti_exploration_distribution(0,1), random_freq_distribution(0.00001,0.99), \
    seed_distribution(0,UINT_MAX), burst_search_distribution(100,UINT_MAX), enable_pre_simplify_distribution(0,1), \
    simplify_delay_distribution(0,UINT_MAX), backtrack_scopes_distribution(100,UINT_MAX), backtrack_conflicts_distribution(2000,UINT_MAX)
    {
        phase_string.push_back("always_false");
        phase_string.push_back("always_true");
        phase_string.push_back("basic_caching");
        phase_string.push_back("random");
        phase_string.push_back("caching");

        restart_string.push_back("static");
        restart_string.push_back("luby");
        restart_string.push_back("ema");
        restart_string.push_back("geometric");

        branching_heuristic_string.push_back("vsids");
        branching_heuristic_string.push_back("chb");

        bool_string.push_back("True");
        bool_string.push_back("False");
    }

    string choose_ints()
    {
        u32 i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11;

        i1 = search_conflicts_distribution(generator);
        i2 = rephase_base_distribution(generator);
        i3 = reorder_base_distribution(generator);
        i4 = reorder_activity_scale_distribution(generator);
        i5 = restart_initial_distribution(generator);
        i6 = variable_decay_distribution(generator);
        i7 = seed_distribution(generator);
        i8 = burst_search_distribution(generator);
        i9 = simplify_delay_distribution(generator);
        i10 = backtrack_scopes_distribution(generator);
        i11 = backtrack_conflicts_distribution(generator);
        return "sat.search.unsat.conflicts=" + to_string(i1) + " sat.search.sat.conflicts=" + to_string(i1) + " sat.rephase.base=" + to_string(i2) + \
        " sat.reorder.base=" + to_string(i3) + " sat.reorder.activity_scale=" + to_string(i4) + " sat.restart.initial=" + to_string(i5) + " sat.variable_decay=" + to_string(i6) + \
        " sat.random_seed=" + to_string(i7) + " sat.burst_search" + to_string(i8) + " sat.simplify.delay=" + to_string(i9) + " sat.backtrack.scopes=" + to_string(i10) + \
        " sat.backtrack.conflicts=" + to_string(i11) + " ";
    }
    
    string choose_doubles()
    {
        double d1, d2, d3, d4, d5;
        d1 = restart_factor_distribution(generator);
        d2 = restart_margin_distribution(generator);
        d3 = restart_emafastglue_distribution(generator);
        d4 = restart_emaslowglue_distribution(generator);
        d5 = random_freq_distribution(generator);
        return "sat.restart.factor=" + to_string(d1) + " sat.restart.margin=" + to_string(d2) + " sat.restart.emafastglue=" + to_string(d3) + " sat.restart.emaslowglue=" + to_string(d4) + " sat.random_freq=" + to_string(d5);
    }

    string choose_strings()
    {
        u32 s1, s2, s3, s4, s5;
        s1 = phase_distribution(generator);
        s2 = restart_distribution(generator);
        s3 = branching_heuristic_distribution(generator);
        s4 = branching_anti_exploration_distribution(generator);
        s5 = enable_pre_simplify_distribution(generator);

        return "sat.phase=" + phase_string[s1] + " sat.restart=" + restart_string[s2] + " sat.branching.heuristic=" + branching_heuristic_string[s3] + " sat.branching.anti_exploration=" + bool_string[s4] + " sat.enable_pre_simplify=" + bool_string[s5] + " ";
    }

    string create_params()
    {
        string int_params = choose_ints();
        string double_params = choose_doubles();
        string string_params = choose_strings();

        return string_params + int_params + double_params;
    }
};

/****************************************************************
 * Define Functions
 ****************************************************************/
void launch_solver(u32 proc_num);
void initial_solvers(u64 nsolvers);
void relaunch_solvers(u64 nsolvers);
void check_solvers();
void prune();

