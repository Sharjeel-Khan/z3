#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>
#include <sys/types.h>
#include <unistd.h>
#include <assert.h>
#include <string>
#include <iterator>
#include <random>
#include <getopt.h>
#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <ensmallen.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;


#ifdef DEBUG
static ofstream logFile("training.log", ios::out);
#define LOG(str) do { logFile << _gettime() << " TRACE: " << __FUNCTION__ << " " << str; cout.flush(); } while( false )
#define LOG_NO_TIME(str) do { cout << str; cout.flush(); } while( false )
#else
#define LOG(str) do { } while ( false )
#define LOG_NO_TIME(str) do { } while ( false )
#endif

static struct option long_options[] =
{
    {"batchSize", optional_argument, NULL, 'b'},
    {"dataFile", optional_argument, NULL, 'd'},
    {"epoch", optional_argument, NULL, 'e'},
    {"hidden", optional_argument, NULL, 'h'},
    {"modelFile", optional_argument, NULL, 'm'},
    {"outputFile", required_argument, NULL, 'o'},
    {"predFile", optional_argument, NULL, 'p'},
    {"rho", optional_argument, NULL, 'r'},
    {"stepSize", optional_argument, NULL, 's'},
    {"testFile", optional_argument, NULL, 't'},
    {NULL, 0, NULL, 0}
};

/****************************************************************
 * Types
 ****************************************************************/
typedef unsigned long long int u64;
typedef unsigned long int u32;

/****************************************************************
 * Global Variables
 ****************************************************************/
u32 batchSize = 16;
u32 epoch = 10;
u32 hidden = 10;
u32 rho = 5;
double stepSize = 1e-4;

string dataFile;
string modelFile;
string outputFile;
string testFile;
string predFile;

bool train, load, pred, output = false;

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