// Various utility functions used by new_test


#ifndef _UTILS_H_
#define _UTILS_H_

#include <cci.h>
#include <string.h>
#include <stdint.h>

#include <fstream>


// Start exactly one daemon process per host
// Returns 0 on success, negated ERRNO on failure
// (Negated because we use positive values for CCI_STATUS codes elsewhere)
int startOneDaemon( char * const * daemonArgs);

// Set up the CCI connection
// Returns a cci_status value, or a negated errno value (ie: -22 for EINVAL)
// If endpointFD is non-NULL, the function will return file descriptor that can
//  block waiting for progress on the endpoint
int initIo( void *buffer, uint64_t len, uint32_t rank, cci_os_handle_t *endpointFd=NULL); 

// Shut down the CCI connection
// Note: doesn't touch the daemon.  (Presumably, it will shut itself down.)
// Returns cci_status or negated errno (ie -22 for EINVAL)
int finalizeIo(void);


// Write the data to a local file
int writeLocal(void *buf, std::streamsize len, std::ofstream &outf);

// Write to the remote daemon (either the GPU or system ram depending
// on what the daemon tells us).
// returns a CCI_STATUS
int writeRemote(void *buf, size_t len, size_t offset, size_t *bytesWritten);

// Get the cache usage data from the daemon
// returns a CCI_STATUS.  
int checkCacheUsage(bool *isEmpty);


#endif // _UTILS_H_
