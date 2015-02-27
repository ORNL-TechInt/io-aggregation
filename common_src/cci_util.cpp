// A few useful utility functions related to CCI
//
// (Right now, there's just one macro for printing debug messages, but I expect
// it will grow.)

#include <cci.h>

#include <iostream>
#include <string>
using namespace std;

void cciDbgMsgLong( const string &func, const string &cciFunc, int ret)
{   
    cerr << func << ": " << cciFunc << " failed with "
         << cci_strerror( NULL,(cci_status)ret) << endl;
}
