#!/usr/bin/env python

# Quick little program to validate the checksums in the output files of
# new_test and daemon
#
# Note: The C++ version is almost 20X faster.  Use it unless there's a very
# good reason not to.
#
# Usage:  validate.py <file1> [ <file2> ... ]
#


import sys
import struct

BLOCK_SIZE=1024

def main():
    if len(sys.argv) < 2:
        print "Usage:", sys.argv[0], "<file1> [ <file2> ... ]"
        return
    
    print "Input files:",
    for fname in sys.argv[1:]:
        print fname,
    print
    
    for fname in sys.argv[1:]:
        validate_file( fname)
                
def validate_file( fname):
    print fname, ":",
    f = open(fname, 'rb')
    checksum = 0;
    address = 0
    while True:
        val_str = f.read(BLOCK_SIZE)
        if len(val_str) == 0:
            print "validated"
            return  # normal exit
        elif len(val_str) < BLOCK_SIZE:
            print "Skipping last", len(val_str), "bytes."
            return
        
        checksum = 0
        format_str = "%dI" % (BLOCK_SIZE/4)
        vals = struct.unpack_from(format_str, val_str)
        if (len(vals) != BLOCK_SIZE/4):
            print "Unexpected block len: ", len(vals)

        for n in vals[:-1]:
            checksum += n
        
        checksum = checksum % 0x100000000
        if checksum != vals[-1]:
            print ": Checksum error for block at 0x%x (0x%x)" % (address, address+BLOCK_SIZE-4)
            print "Calculated checksum: %d   File's checksum: %d" %(checksum, vals[-1])
            return  
#            else:
#                print "checksum pass for block at", address
        address += BLOCK_SIZE




if __name__ == "__main__":
    main()
