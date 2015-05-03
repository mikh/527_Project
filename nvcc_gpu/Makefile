GCC = g++
CC = nvcc
CLAGS = -Wall -lm 
#`pkg-config --cflags opencv`
INCLUDE = -I/usr/include/opencv -I/usr/include/opencv2
LDFLAGS = $(INCLUDE) -L/usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_highgui -lopencv_imgproc `pkg-config --libs opencv` -lrt

ALL = neural_read_gpu

all: $(ALL)

neural_write: main_split_save_vectors_gpu.o NeuralNet.o ImageData.o
	$(CC) $(CFLAGS) -o neural_write main_split_save_vectors_gpu.o NeuralNet.o ImageData.o $(LDFLAGS)

neural_read_gpu: main_split_load_vectors_gpu.o NeuralNet.o
	$(CC) $(CFLAGS) -o neural_read_gpu main_split_load_vectors_gpu.o NeuralNet.o -lrt

main_split_save_vectors_gpu.o: main_split_save_vectors_gpu.cu
	$(CC) $(CFLAGS) $(INCLUDE) -c main_split_save_vectors_gpu.cu

main_split_load_vectors_gpu.o: main_split_load_vectors_gpu.cu
	$(CC) $(CFLAGS) -c main_split_load_vectors_gpu.cu -lrt

NeuralNet.o: Neuron.h Layer.h NeuralNet.cpp NeuralNet.h
	$(GCC) $(CFLAGS) -c Neuron.h NeuralNet.cpp

ImageData.o: ImageData.cpp ImageData.h
	$(GCC) $(CFLAGS) $(INCLUDE) -c ImageData.cpp

.PHONY: clean

clean:
	rm -rf *.o *.gch $(ALL)
