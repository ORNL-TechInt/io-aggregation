#!/bin/env python
#
# A script for analyzing the client-side timing data from the io-agg
# 'new_test' executable
#
# Sample output from a single rank (from a small run):
#
# len 134217728 start 1426876784488692 end 1426876784674608 elapsed 185916 MB/s 688.483
# len 134217728 start 1426876796307678 end 1426876796485084 elapsed 177406 MB/s 721.509
# len 134217728 start 1426876805916297 end 1426876806085703 elapsed 169406 MB/s 755.581
# len 268435456 start 1426876818203246 end 1426876818585501 elapsed 382255 MB/s 669.71
# len 268435456 start 1426876969876948 end 1426876970275177 elapsed 398229 MB/s 642.846
# len 268435456 start 1426877063784120 end 1426877064170422 elapsed 386302 MB/s 662.694
# len 536870912 start 1426877086479164 end 1426877087494383 elapsed 1015219 MB/s 504.325
# len 536870912 start 1426877129710784 end 1426877130458838 elapsed 748054 MB/s 684.443
# len 536870912 start 1426877166373242 end 1426877167339400 elapsed 966158 MB/s 529.934
#
#

import os
import os.path
import sqlite3

from optparse import OptionParser

import plot_utils as pu

#Plot function

def timelines(y, xstart, xstop, color='b'):
    """Plot timelines at y from xstart to xstop with given color."""   
    plt.hlines(y, xstart, xstop, color, lw=4)
    plt.vlines(xstart, y+0.03, y-0.03, color, lw=2)
    plt.vlines(xstop, y+2, y-2, color, lw=2)


TABLE_NAME = 'Results'

def init_db( db_filename):
    '''Initialize the database
    
    Opens the specified file, drops any existing tables and creates new ones.
    '''
    conn = sqlite3.connect(db_filename)
    c = conn.cursor()
    
    c.execute( "DROP TABLE IF EXISTS %s;"%TABLE_NAME)
    
    # Create table
    stmt = "CREATE TABLE %s ("                                \
        "rank int, iteration int, node int, length int, "     \
        "start int, end int, elapsed float, bw int);" % TABLE_NAME
    c.execute(stmt)
    conn.commit()
    
    return conn
    
def import_results( filename, rank, ranks_per_node, conn):
    '''Open the specifed file and read its contents into the DB
    '''
    
    f = open( filename, mode='r')
    if (not f):
        return []
    
    l= []
    lines = f.readlines()
    for i in range( len(lines)):
        
        parts = lines[i].split()
        length = int(parts[1])
        start = int(parts[3])
        end = int(parts[5])
        elapsed = float(end - start) / float(1000 * 1000) # time in seconds
        bw = int(length / elapsed) # bandwidth in bytes/sec
        stmt = "INSERT INTO %s VALUES (?, ?, ?, ?, ?, ?, ?, ?);"%TABLE_NAME
        c = conn.cursor()
        
        node = rank / ranks_per_node
        # Note: We're relying on the fact that lays out ranks sequentialy:
        # If ranks_per_node is 8, ranks 0-7 are on node 0, 8-15 are on
        # node 1, etc..
        
        c.execute( stmt, (rank, i, node, length, start, end, elapsed, bw))
        conn.commit()

        
def main():

    parser = OptionParser()
    
    parser.add_option("-d", "--data_dir",
                      type="string", dest="data_dir",
                      help="directory where the data files reside",
                      metavar="DIR")
    
    parser.add_option("-N", "--ranks_per_node",
                      type="int", dest="ranks_per_node",
                      help="number of ranks that ran on each node")
    
    parser.add_option("-D", "--db_file",
                      type="string", dest="db_file",
                      help="name of an existing database file",
                      metavar="FILE")
    

    (options, args) = parser.parse_args()
    if (not options.data_dir) and (not options.db_file):
        parser.error('Either data dir or database file must be specified')
    if options.data_dir and options.db_file:
        parser.error('--data_dir and --db_file are mutually exclusive')
    if options.data_dir and not options.ranks_per_node:
        parser.error('Ranks per node not specified')
    try:
        if options.db_file:
            open( options.db_file)
    except IOError:
        parser.error( "Could not open db_file: %s"%options.db_file)
    
    if options.data_dir:
        options.data_dir = os.path.abspath(options.data_dir)
        # Get rid of any . or .. parts of the path...
    
    
    # Done handling command line args.  On to the main part of the code...

    
    if options.data_dir:
        # DB filename will be based on the data_dir's name
        conn = init_db( "%s.db"%os.path.basename(options.data_dir))
        
        # Read all the client data into the DB
        print "Importing data files."
        files = os.listdir( options.data_dir)
        for f in files:
            if f[0:5] == 'rank-' and \
            len(f) == 9:
                rank = int(f[5:9])
                print "%s..."%f,
                import_results( options.data_dir + os.sep + f, rank,
                                options.ranks_per_node, conn)
                print "Done"
    else:
        conn = sqlite3.connect( options.db_file)
        # connect to the existing db
              

    pu.plot_per_node_bw_non_gpu( conn, 2)
    pu.plot_per_rank_bw( conn, 2)
    pu.plot_per_node_bw( conn, 2)

if __name__ == "__main__":
    main()
