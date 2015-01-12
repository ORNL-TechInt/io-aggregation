ifndef CCI
$(error Export path to CCI in variable named CCI)
endif

ifndef MPICC
ifndef MPI
$(error Export path to MPI in variable named MPI)
endif
MPICC = $(MPI)/bin/mpicc
endif

CC = gcc
CFLAGS = -g -O0 -std=c99 -D_XOPEN_SOURCE=600 -Wall -pedantic -I$(CCI)/include -fPIC
LDFLAGS = -dynamic -L$(CCI)/lib -lcci -lpthread
OBJS = io.o

C_OBJS = iod.o
C_TARGETS = iod

MPI_OBJS = test.o
MPI_TARGETS = test

ALL:$(OBJS) $(C_OBJS) $(MPI_OBJS) $(C_TARGETS) $(MPI_TARGETS)
$(C_OBJS):%.o:%.c io.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(MPI_OBJS):%.o:%.c io.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(C_TARGETS):$(C_OBJS) $(OBJS)
	$(CC) $(OBJS) $(C_OBJS) $(LDFLAGS) -o $@

$(MPI_TARGETS):$(MPI_OBJS) $(OBJS)
	$(MPICC) $(OBJS) $(MPI_OBJS) $(LDFLAGS) -o $@

clean:
	rm -rf $(C_TARGETS) $(MPI_TARGETS) $(C_OBJS) $(MPI_OBJS) $(OBJS) *.dSYM
