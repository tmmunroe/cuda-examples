SDK_INSTALL_PATH := /usr/local/cuda-11.0
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64 -lcudnn -lcublas -lcublasLt -lz
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 -g -G
OPTIONS   :=  -O3
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := cuda-examples.tar
EXECS :=  vecadd00 matmult00 vecadd01 matmult01 matmult02 arrayadd arrayaddUnifiedMemory conv
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile README.md *.h *.cu *.pdf *.txt *.py
#######################################################################

timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#######################################################################
vecaddKernel00.o : vecaddKernel00.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

vecadd00 : vecadd.cu vecaddKernel.h vecaddKernel00.o timer.o
	${NVCC} $< vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

#
# used the global thread id (computed from block dim and threadIdx) to index into
#  vectors being added.. this ensured threads in the same warp (as ordered by threadId)
#  access contiguous memory locations
vecaddKernel01.o : vecaddKernel01.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

vecadd01 : vecadd.cu vecaddKernel.h vecaddKernel01.o timer.o
	${NVCC} $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
### vecaddKernel01.o : vecaddKernel01.cu
###	${NVCC} $< -c -o $@ $(OPTIONS)
###
### vecadd01 : vecadd.cu vecaddKernel.h vecaddKernel01.o timer.o
###	${NVCC} $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
## Provided Kernel
matmultKernel00.o : matmultKernel00.cu matmultKernel.h 
	${NVCC} $< -c -o $@ $(OPTIONS)

matmult00 : matmult.cu  matmultKernel.h matmultKernel00.o timer.o
	${NVCC} $< matmultKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
## Expanded Kernel, notice that FOOTPRINT_SIZE is redefined (from 16 to 32)
matmultKernel01.o : matmultKernel01.cu matmultKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS) -DFOOTPRINT_SIZE=32

matmult01 : matmult.cu  matmultKernel.h matmultKernel01.o timer.o
	${NVCC} $< matmultKernel01.o -o $@ $(LIB) timer.o $(OPTIONS) -DFOOTPRINT_SIZE=32



#######################################################################
## Expanded Kernel, notice that FOOTPRINT_SIZE is redefined (from 16 to 32)
matmultKernel02.o : matmultKernel02.cu matmultKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS) -DFOOTPRINT_SIZE=32

matmult02 : matmult.cu  matmultKernel.h matmultKernel02.o timer.o
	${NVCC} $< matmultKernel02.o -o $@ $(LIB) timer.o $(OPTIONS) -DFOOTPRINT_SIZE=32


#######################################################################
## arrayadd program
arrayaddKernel.o : arrayaddKernel.cu arrayaddKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS) -DFOOTPRINT_SIZE=32

arrayadd : arrayadd.cu arrayaddKernel.h arrayaddKernel.o timer.o
	${NVCC} $< arrayaddKernel.o -o $@ $(LIB) timer.o $(OPTIONS)

arrayaddUnifiedMemory : arrayaddUnifiedMemory.cu arrayaddKernel.h arrayaddKernel.o timer.o
	${NVCC} $< arrayaddKernel.o -o $@ $(LIB) timer.o $(OPTIONS)

#######################################################################
## conv program
conv : conv.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)

