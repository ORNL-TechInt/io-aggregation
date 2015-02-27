// Various utility functions used by new_test


#ifndef _UTILS_H_
#define _UTILS_H_

#include <string.h>
#include <stdint.h>

// Start up the daemon and set up the CCI connection
// Returns a cci_status value, or a negated errno value (ie: -22 for EINVAL)
int initIo( void *buffer, uint32_t len, uint32_t rank,
            uint32_t localRanks, char **daemon_args);

// Shut down the CCI connection
// Note: doesn't touch the daemon.  (Presumably, it will shut itself down.)
// Returns cci_status or negated errno (ie -22 for EINVAL)
int finalizeIo(void);


#endif // _UTILS_H_