ifndef CCI
$(error Export path to CCI in variable named CCI)
endif

ifndef MPICC
ifndef MPI
$(error Export path to MPI in variable named MPI)
endif
MPICC = $(MPI)/bin/mpicc
endif

ifndef CUDA_HOME
$(warning CUDA_HOME not defined.  Removing gpu_caching_iod target from 'ALL'.)
$(warning Export path to CUDA installation in variable named CUDA_HOME to enable.)
else
GPU_CACHING_IOD=gpu_caching_iod
endif

CC = gcc
NVCC = $(CUDA_HOME)/bin/nvcc
CFLAGS = -g -O3 -std=c99 -D_XOPEN_SOURCE=600 -Wall -pedantic -I$(CCI)/include -fPIC
NVCFLAGS = -g -O0 -Xcompiler -std=c99 -D_XOPEN_SOURCE=600 -Xcompiler -Wall -Xcompiler -pedantic -I$(CCI)/include -Xcompiler -fPIC
LDFLAGS = -dynamic -L$(CCI)/lib -lcci -lpthread -Wl,-rpath,$(CCI)/lib
NVLDFLAGS = -L$(CCI)/lib -lcci -lpthread -Xlinker -rpath=$(CCI)/lib
OBJS = io.o

C_OBJS = iod.o
C_TARGETS = iod

MPI_OBJS = test.o
MPI_TARGETS = test

CACHING_IOD_OBJS = caching_iod.o
GPU_CACHING_IOD_OBJS = gpu_caching_iod.o

ALL:$(OBJS) $(C_OBJS) $(MPI_OBJS) $(C_TARGETS) $(MPI_TARGETS) caching_iod $(GPU_CACHING_IOD)
$(OBJS):%.o:%.c io.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(C_OBJS):%.o:%.c io.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(MPI_OBJS):%.o:%.c io.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(CACHING_IOD_OBJS):%.o:%.c io.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(GPU_CACHING_IOD_OBJS):gpu_%.o:%.c io.h
	$(NVCC) $(NVCFLAGS) $(CPPFLAGS) -c $< -o $@

$(C_TARGETS):$(C_OBJS) $(OBJS)
	$(CC) $(OBJS) $(C_OBJS) $(LDFLAGS) -o $@

$(MPI_TARGETS):$(MPI_OBJS) $(OBJS)
	$(MPICC) $(OBJS) $(MPI_OBJS) $(LDFLAGS) -o $@

caching_iod: $(CACHING_IOD_OBJS) $(OBJS)
	$(CC) $(OBJS) $(CACHING_IOD_OBJS) $(LDFLAGS) -o $@

gpu_caching_iod: $(GPU_CACHING_IOD_OBJS) $(OBJS)
	$(NVCC) $(OBJS) $(GPU_CACHING_IOD_OBJS) $(NVLDFLAGS) -o $@

clean:
	rm -rf $(C_TARGETS) $(MPI_TARGETS) caching_iod gpu_caching_iod $(C_OBJS) $(MPI_OBJS) $(CACHING_IOD_OBJS) $(GPU_CACHING_IOD_OBJS) $(OBJS) *.dSYM
