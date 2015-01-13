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
LDFLAGS = -dynamic -L$(CCI)/lib -lcci -lpthread -Wl,-rpath,$(CCI)/lib
OBJS = io.o

C_OBJS = iod.o
C_TARGETS = iod

MPI_OBJS = test.o
MPI_TARGETS = test

CACHING_IOD_OBJS = caching_iod.o

ALL:$(OBJS) $(C_OBJS) $(MPI_OBJS) $(C_TARGETS) $(MPI_TARGETS) caching_iod
$(OBJS):%.o:%.c io.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(C_OBJS):%.o:%.c io.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(MPI_OBJS):%.o:%.c io.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(CACHING_IOD_OBJS):%.o:%.c io.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(C_TARGETS):$(C_OBJS) $(OBJS)
	$(CC) $(OBJS) $(C_OBJS) $(LDFLAGS) -o $@

$(MPI_TARGETS):$(MPI_OBJS) $(OBJS)
	$(MPICC) $(OBJS) $(MPI_OBJS) $(LDFLAGS) -o $@

caching_iod: $(CACHING_IOD_OBJS) $(OBJS)
	$(CC) $(OBJS) $(CACHING_IOD_OBJS) $(LDFLAGS) -o $@

clean:
	rm -rf $(C_TARGETS) $(MPI_TARGETS) caching_iod $(C_OBJS) $(MPI_OBJS) $(CACHING_IOD_OBJS) $(OBJS) *.dSYM
