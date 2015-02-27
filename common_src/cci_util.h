// A few useful utility functions related to CCI
//
// (Right now, there's just one macro for printing debug messages, but I expect
// it will grow.)

#ifndef _CCI_UTIL_H_
#define _CCI_UTIL_H_

#include <string>

// Ideally, this definition would be hidden somehow.  The macro is "public" interface...
void cciDbgMsgLong( const std::string &func, const std::string &cciFunc, int ret);

#define cciDbgMsg( cciFunc, ret) \
    cciDbgMsgLong( __func__, cciFunc, ret);

    
#endif // _CCI_UTIL_H_
