// Implementation file for the IoRequest class


#include "iostats.h"

#include <ostream>
using namespace std;

std::ostream &IoStats::writeResults( ostream &outf)
{
    return outf
        << "Actual_Len "     << m_actualLen
        << " Requested_Len " << m_requestedLen
        << " Recv "          << m_recv
        << " Reply "         << m_reply
        << " Enqueue "       << m_enqueue
        << " Dequeue "       << m_dequeue
        << " Done "          << m_done
        << endl;
}