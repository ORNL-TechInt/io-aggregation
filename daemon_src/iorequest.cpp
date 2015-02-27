// Implementation file for the IoRequest class


#include "iorequest.h"

#include <ostream>
using namespace std;


// The output line looks like:
// len 1048576 rx_us 1425063324760922 cpy_us 0 rma_us 0 deq_us 1425063324762017 io_us 1425063324762541

std::ostream &IoRequest::writeResults( ostream &outf)
{
    return outf
        << "len "     << len
        << " rx_us "  << rx_us
        << " cpy_us " << cpy_us
        << " rma_us " << rma_us
        << " deq_us " << deq_us
        << " io_us "  << io_us
        << endl;
}