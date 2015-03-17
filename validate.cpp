// A quick little program that will validate the checksums new_test and
// daemon write into their data files.
//
// NOTE: The algorithm this code uses must obviously match the algorithm
// that the client uses!  See initBuffer() in new_test.cpp!

#include <stdint.h>

#include <fstream>
#include <iostream>
#include <string>
using namespace std;


void validateFile( const string &fname)
{

    ifstream infile(fname.c_str(), ios_base::binary | ios_base::ate);
    streampos fileSize = infile.tellg();
    infile.seekg( 0);
    
    if (!infile) {
        cerr << "Failed to open " << fname << ".  Continuing with next file." << endl;
        return;
    } else {
        cout << fname << ": " << flush;
    }
    
    
    uint32_t vals[256];
    uint32_t checksum = 0;
    uint64_t address = 0;
    
    while (true) {
        streampos startPos = infile.tellg();
        infile.read( (char *)vals, 1024);
               
        if (infile.eof()) {
            if (fileSize != startPos) {
                cout << "Skipping last " << fileSize - startPos << " bytes. " << endl;
            } else {
                cout << "validated" << endl;
            }
            return;
        }
        
        checksum = 0;
        for (unsigned i = 0; i < 255; i++) {
            checksum += vals[i];
        }
        if (vals[255] != checksum) {
            cout << "Checksum error at 0x" << hex << address
                    << ".  Calculated checksum: " << dec << checksum
                    << "   File's checksum: " << vals[255] << endl;
            return;
        }
        
        address += 1024;
    }
    
    cerr << "BUG! This line should never be printed!" << endl;
    return;
}
 
 
 
int main( int argc, char **argv)
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << "<filename> [<filename> ...]" << endl;
        return 1;
    }
    
    for (int i=1; i < argc; i++) {
        validateFile( argv[i]);
    }
    
    return 0;
}