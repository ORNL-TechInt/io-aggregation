ifndef CCI
$(error Export path to CCI in variable named CCI)
endif

ifndef MPICC
ifndef MPI
$(error Export path to MPI in variable named MPI)
endif
MPICC = $(MPI)/bin/mpic++
endif

ifndef CUDA_HOME
$(warning CUDA_HOME not defined.  Removing gpu_caching_iod target from 'ALL'.)
$(warning Export path to CUDA installation in variable named CUDA_HOME to enable.)
else
GPU_CACHING_IOD=gpu_caching_iod
endif

#CC = g++ # use MPICC instead...

VPATH = common_src:client_src:daemon_src

NVCC = $(CUDA_HOME)/bin/nvcc
CFLAGS = -g -O0 -D_XOPEN_SOURCE=600 -std=c++0x -Wall -pedantic -I./common_src -I$(CCI)/include
NVCFLAGS = -g -O0 -D_XOPEN_SOURCE=600 -Xcompiler -std=c++0x -Xcompiler -Wall -Xcompiler -pedantic -I./common_src -I$(CCI)/include -Xcompiler -fPIC

LDFLAGS = -dynamic -L$(CCI)/lib -lcci -lpthread -Wl,-rpath,$(CCI)/lib
NVLDFLAGS = -L$(CCI)/lib -lcci -lpthread -Xlinker -rpath=$(CCI)/lib


DAEMON_DEPS = cacheblock.h cci_msg.h cci_util.h daemoncmdlineopts.h iorequest.h peer.h timing.h
DAEMON_SRC = cci_util.cpp daemon.cpp daemoncmdlineopts.cpp iorequest.cpp peer.cpp

NEW_TEST_DEPS = cci_msg.h cci_util.h cmdlineopts.h timing.h utils.h
NEW_TEST_SRC = cci_util.cpp new_test.cpp utils.cpp cmdlineopts.cpp




NEW_TEST_OBJS = $(patsubst %.cpp,obj/new_test/%.o,$(NEW_TEST_SRC))
DAEMON_OBJS = $(patsubst %.cpp,obj/daemon/%.o,$(DAEMON_SRC))


ALL: new_test daemon

new_test: $(NEW_TEST_OBJS)
	$(MPICC) -o $@ $^ $(LDFLAGS)

daemon: $(DAEMON_OBJS)
	$(MPICC) -o $@ $^ $(LDFLAGS)

$(NEW_TEST_OBJS):obj/new_test/%.o: %.cpp $(NEW_TEST_DEPS)
	$(MPICC) -c -o $@ $< $(CFLAGS)

$(DAEMON_OBJS):obj/daemon/%.o: %.cpp $(DAEMON_DEPS)
	$(MPICC) -c -o $@ $< $(CFLAGS)

clean:
	rm -f new_test $(NEW_TEST_OBJS) daemon $(DAEMON_OBJS) *.dSYM
