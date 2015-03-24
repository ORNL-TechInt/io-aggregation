# Utility functions for making the plots we want
# (Should probably be a python package...)

import matplotlib.pyplot as plt
import sqlite3
import numpy as np

#Plot function

def timelines(y, xstart, xstop, color='b'):
    """Plot timelines at y from xstart to xstop with given color."""   
    plt.hlines(y, xstart, xstop, color, lw=4)
    plt.vlines(xstart, y+0.03, y-0.03, color, lw=2)
    plt.vlines(xstop, y+0.03, y-0.03, color, lw=2)





TABLE_NAME="Results"

# Plot average per-rank bandwidth as a function of write size
# Assumes:
#  First n/2 ranks used FCFS allocator, second n/2 ranks used balanced allocator
def plot_per_rank_bw( conn, num_write_threads):
    
    c = conn.cursor()
    c.execute( "SELECT max(rank) from %s;"%TABLE_NAME)
    max_rank = c.fetchall()[0][0]
    
    # Queries that we'll execute (the plot will show 2 groupings)
    statements = ["SELECT length, bw from %s WHERE rank < %d ORDER BY length;" \
                  %(TABLE_NAME, max_rank/2),
                  "SELECT length, bw from %s WHERE rank >= %d ORDER BY length;" \
                  %(TABLE_NAME, max_rank/2) ]
    
    rects = [] # the Rectangle instances returned by calls to bar() below
    fig, ax = plt.subplots() # initialize the plot
    width = 0.35                # the width of the bars
    colors = ['r','b']   # colors of the bars for the different groups
    
    for i in range( len(statements)):
        c.execute( statements[i])
        data=c.fetchall()
        
        prev_len = 0
        length_dict = { }  # keys are the write lengths, values are numpy
                        # arrays of the BW values
        for row in data:
            if row[0] != prev_len:
                length_dict[row[0]] = []
                prev_len = row[0]
            length_dict[row[0]].append( row[1])
            # NumPy arrays don't have append(), so start with a list
            # (we'll convert to an array later)
        
        for k in length_dict.keys():
            length_dict[k] = np.array(length_dict[k], float)
        
        # now have a dictionary of numpy arrays
        # convert values from bytes/sec to MB/s and keys from bytes
        # to MB
        MB=1024*1024
        for k in length_dict.keys():
            length_dict[k/MB] = length_dict[k] / MB
            length_dict.pop(k)
            
        # Get the min, max & average values
        # Note: we can't count on the order of items in a
        # dictionary, so we're using lists
        x_vals = length_dict.keys()
        x_vals.sort()
        mins = []
        maxes = []
        averages = []
        for k in x_vals:
            mins.append( np.amin( length_dict[k]))
            maxes.append( np.amax( length_dict[k]))
            averages.append( np.average( length_dict[k]))
        
        
        # debugging:
        print "Mins:", mins
        print "Maxes:", maxes
        print "Averages:", averages
        
        # plot the data
        ind = np.arange(len(mins))  # the x locations for the groups
        rects.append( ax.bar(ind + (width*i), averages, width, color=colors[i]))
    
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Bandwidth (MB/s)')
    ax.set_title('Per-Rank Bandwidth vs. Write Size')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( x_vals)
    ax.set_xlabel( 'Write size (MB)')
    ax.legend( (rects[0][0], rects[1][0]), ('First Come/First Served', 'Balanced') )

#    def autolabel(rects):
#        # attach some text labels
#        for rect in rects:
#            height = rect.get_height()
#            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
#                    ha='center', va='bottom')

    #autolabel(rects1)
    #autolabel(rects2)

    plt.show()
    
    
# Similar to plot_per_rank_bw() above, but we sum all the bw values
# for each node and report on that sum.
#
# Assumes:
#  First n/2 nodes used FCFS allocator, second n/2 nodes used balanced allocator
def plot_per_node_bw( conn, num_write_threads):
    
    c = conn.cursor()
    c.execute( "SELECT max(rank) from %s;"%TABLE_NAME)
    max_rank = c.fetchall()[0][0]
    
    # Queries that we'll execute (the plot will show 2 groupings)
    statements = ["SELECT length, SUM(bw) from %s WHERE rank < %d GROUP BY node, iteration ORDER BY length;" \
                  %(TABLE_NAME, max_rank/2),
                  "SELECT length, SUM(bw) from %s WHERE rank >= %d GROUP BY node, iteration ORDER BY length;" \
                  %(TABLE_NAME, max_rank/2) ]
    # The GROUP BY & SUM() clauses cause us to sum the bw value for all rows
    # with identical values for the (node, iteration) pair.  Since iteration
    # increments on every write(), the only rows with identical (node,iteration)
    # values will be the ones for different ranks.  (ie: no need to include
    # length in the GROUP BY clause)
    
    rects = [] # the Rectangle instances returned by calls to bar() below
    fig, ax = plt.subplots() # initialize the plot
    width = 0.35                # the width of the bars
    colors = ['r','b']   # colors of the bars for the different groups
    
    for i in range( len(statements)):
        c.execute( statements[i])
        data=c.fetchall()
        
        prev_len = 0
        length_dict = { }  # keys are the write lengths, values are numpy
                        # arrays of the BW values
        for row in data:
            if row[0] != prev_len:
                length_dict[row[0]] = []
                prev_len = row[0]
            length_dict[row[0]].append( row[1])
            # NumPy arrays don't have append(), so start with a list
            # (we'll convert to an array later)
        
        for k in length_dict.keys():
            length_dict[k] = np.array(length_dict[k], float)
        
        # now have a dictionary of numpy arrays
        # convert values from bytes/sec to MB/s and keys from bytes
        # to MB
        MB=1024*1024
        for k in length_dict.keys():
            length_dict[k/MB] = length_dict[k] / MB
            length_dict.pop(k)
            
        # Get the min, max & average values
        # Note: we can't count on the order of items in a
        # dictionary, so we're using lists
        x_vals = length_dict.keys()
        x_vals.sort()
        mins = []
        maxes = []
        averages = []
        for k in x_vals:
            mins.append( np.amin( length_dict[k]))
            maxes.append( np.amax( length_dict[k]))
            averages.append( np.average( length_dict[k]))
        
        
        # debugging:
        print "Mins:", mins
        print "Maxes:", maxes
        print "Averages:", averages
        
        # plot the data
        ind = np.arange(len(mins))  # the x locations for the groups
        rects.append( ax.bar(ind + (width*i), averages, width, color=colors[i]))
    
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Bandwidth (MB/s)')
    ax.set_title('Per-Node Bandwidth vs. Write Size')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( x_vals)
    ax.set_xlabel( 'Write size (MB)')
    ax.legend( (rects[0][0], rects[1][0]), ('First Come/First Served', 'Balanced') )

#    def autolabel(rects):
#        # attach some text labels
#        for rect in rects:
#            height = rect.get_height()
#            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
#                    ha='center', va='bottom')

    #autolabel(rects1)
    #autolabel(rects2)

    plt.show()
    
    
    
    
            
            
    
    