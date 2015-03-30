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
    
    generic_bar_plot( conn, statements, 
                      title = 'Average Per-Rank Bandwidth vs. Write Size',
                      x_label = 'Write size (MB)',
                      y_label = 'Bandwidth (MB/s)',
                      legend=('First Come/First Served', 'Balanced'))
                     
                     
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
    
    generic_bar_plot( conn, statements, 
                      title = 'Average Per-Node Bandwidth vs. Write Size',
                      x_label = 'Write size (MB)',
                      y_label = 'Bandwidth (MB/s)',
                      legend=('First Come/First Served', 'Balanced'))


# Bar chart of per-node BW vs. write size.  Assumes all ranks were
# writing the normal way (not using the GPU cache)
def plot_per_node_bw_non_gpu( conn, num_write_threads):
    # Queries that we'll execute (the plot will show 2 groupings)
    statements = ["SELECT length, SUM(bw) from %s GROUP BY node, iteration ORDER BY length;" \
                  %TABLE_NAME ]
    # See above for comments about the GROUP BY & SUM() clauses
    
    generic_bar_plot( conn, statements, 
                      title = 'Average Per-Node Bandwidth vs. Write Size',
                      x_label = 'Write size (MB)',
                      y_label = 'Bandwidth (MB/s)')
                     
                     
# Bar chart of the per-node BW vs. write size.  Assumes 192 nodes,
# nodes 0-63: 1 write thread, nodes 64-127: 2 write threads and
# nodes 128-191: 4 write threads
def plot_per_node_bw_group_by_write_thread( conn):
    c = conn.cursor()
    #c.execute( "SELECT max(rank) from %s;"%TABLE_NAME)
    #max_rank = c.fetchall()[0][0]
    
    # Queries that we'll execute (the plot will show 3 groupings)
    statements = ["SELECT length, SUM(bw) from %s WHERE node < 64 GROUP BY node, iteration ORDER BY length;"%TABLE_NAME,
                  "SELECT length, SUM(bw) from %s WHERE rank >= 64 and rank < 128 GROUP BY node, iteration ORDER BY length;"%TABLE_NAME,
                  "SELECT length, SUM(bw) from %s WHERE rank < 192 GROUP BY node, iteration ORDER BY length;"%TABLE_NAME]
    # The GROUP BY & SUM() clauses cause us to sum the bw value for all rows
    # with identical values for the (node, iteration) pair.  Since iteration
    # increments on every write(), the only rows with identical (node,iteration)
    # values will be the ones for different ranks.  (ie: no need to include
    # length in the GROUP BY clause)
    
    generic_bar_plot( conn, statements, 
                      title = 'Average Per-Node Bandwidth vs. Write Size',
                      x_label = 'Write size (MB)',
                      y_label = 'Bandwidth (MB/s)',
                      legend=('1 Write Thread', '2 Write Threads',
                              '4 Write Threads'))
    


# Bar chart of the per-RANK BW vs. write size.  Assumes 192 nodes,
# nodes 0-63: 1 write thread, nodes 64-127: 2 write threads and
# nodes 128-191: 4 write threads
def plot_per_rank_bw_group_by_write_thread( conn):
    c = conn.cursor()
    #c.execute( "SELECT max(rank) from %s;"%TABLE_NAME)
    #max_rank = c.fetchall()[0][0]
    
    # Queries that we'll execute (the plot will show 3 groupings)
    statements = ["SELECT length, bw from %s WHERE node < 64 ORDER BY length;"%TABLE_NAME,
                  "SELECT length, bw from %s WHERE rank >= 64 and rank < 128 ORDER BY length;"%TABLE_NAME,
                  "SELECT length, bw from %s WHERE rank < 192 ORDER BY length;"%TABLE_NAME]
    # The GROUP BY & SUM() clauses cause us to sum the bw value for all rows
    # with identical values for the (node, iteration) pair.  Since iteration
    # increments on every write(), the only rows with identical (node,iteration)
    # values will be the ones for different ranks.  (ie: no need to include
    # length in the GROUP BY clause)
    
    generic_bar_plot( conn, statements, 
                      title = 'Average Per-Rank Bandwidth vs. Write Size',
                      x_label = 'Write size (MB)',
                      y_label = 'Bandwidth (MB/s)',
                      legend=('1 Write Thread', '2 Write Threads',
                              '4 Write Threads'))
    
    


# Executes the specified SQL statements and generates a bar plot
# Not exepected to be used directly - call one of the plot_*
# functions instead.
# Note: statements must be a list (or tuple) of strings, even if there's
# only one...
# NOTE: The statements must be constructed such that the independent
# variable is the first value in each row, and the dependent variable
# is the second.
# Legend (if it exists) must also be an iterable
def generic_bar_plot( conn, statements, title=None,
                       x_label=None, y_label=None, legend = None):
    c = conn.cursor()       
    rects = [] # the Rectangle instances returned by calls to bar() below
    fig, ax = plt.subplots() # initialize the plot
    width = 0.35                # the width of the bars
    colors = ['b', 'g', 'r', 'c', 'm', 'y']   # colors of the bars for the different groups
    
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
        
        
        # I want to display min & max values as error bars on
        # each main bar.  MatPlotLib can do this, but it treats
        # the upper & lower error values as values to be added
        # or subtracted from the main value.  Hence the next two
        # lines
        low_err = np.array(averages) - np.array(mins)
        high_err = np.array(maxes) - np.array(averages)
        
        # debugging:
        print "Mins:", mins
        print "Maxes:", maxes
        print "Averages:", averages
        print "Low error:", low_err
        print "High error:", high_err
        
        # plot the data
        ind = np.arange(len(mins))  # the x locations for the groups
        rects.append( ax.bar(ind + (width*i), averages, width,
                             color=colors[i], yerr=[low_err,high_err]))
    
    
    # add some text for labels, title and axes ticks
    ax.set_xticks(ind+width)
    ax.set_xticklabels( x_vals)
    if title:
        ax.set_title( title)
    if x_label:
        ax.set_xlabel( x_label)
    if y_label:
        ax.set_ylabel( y_label)
    if legend:
        # TODO: figure out how to properly build the tuple
        # of rects[n][0]...
        artists = []
        for r in rects:
            artists.append(r[0])
            # 'artist' is the term used in the docs for legend()...
        ax.legend( artists, legend)

#    def autolabel(rects):
#        # attach some text labels
#        for rect in rects:
#            height = rect.get_height()
#            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
#                    ha='center', va='bottom')

    #autolabel(rects1)
    #autolabel(rects2)

    plt.show()

    
            
            
    
    