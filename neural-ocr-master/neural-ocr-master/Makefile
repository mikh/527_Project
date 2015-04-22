CC = g++
CLAGS = -Wall -lm `pkg-config --cflags opencv`
INCLUDE = -I/usr/include/opencv -I/usr/include/opencv2
LDFLAGS = $(INCLUDE) -L/usr/lib/x86_64-linux-gnu \
          -lopencv_core -lopencv_highgui -lopencv_imgproc `pkg-config --libs opencv`
ALL = neural

all: $(ALL)

neural: main.o NeuralNet.o ImageData.o
	$(CC) $(CFLAGS) -o neural main.o NeuralNet.o ImageData.o $(LDFLAGS)

main.o: main.cpp
	$(CC) $(CFLAGS) -Wno-write-strings $(INCLUDE) -c main.cpp

NeuralNet.o: Neuron.h Layer.h NeuralNet.cpp NeuralNet.h
	$(CC) $(CFLAGS) -c Neuron.h NeuralNet.cpp

ImageData.o: ImageData.cpp ImageData.h
	$(CC) $(CFLAGS) $(INCLUDE) -c ImageData.cpp

.PHONY: clean

clean:
	rm -rf *.o *.gch $(ALL)
