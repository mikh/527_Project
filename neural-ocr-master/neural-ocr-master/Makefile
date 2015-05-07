CC = g++
CLAGS = -Wall -lm 
#`pkg-config --cflags opencv`
INCLUDE = -I/usr/include/opencv -I/usr/include/opencv2
LDFLAGS = $(INCLUDE) -L/usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_highgui -lopencv_imgproc `pkg-config --libs opencv`
#INCLUDE = -I/home/mikh/EC527/cflags/cflags/opencv -I/home/mikh/EC527/cflags/cflags/opencv2
#LDFLAGS = $(INCLUDE) -L /home/mikh/EC527/libs/libs -lopencv_core -lopencv_highgui -lopencv_imgproc /home/mikh/EC527/libs/libs/libopencv_calib3d.so /home/mikh/EC527/libs/libs/libopencv_contrib.so /home/mikh/EC527/libs/libs/libopencv_core.so /home/mikh/EC527/libs/libs/libopencv_features2d.so /home/mikh/EC527/libs/libs/libopencv_flann.so /home/mikh/EC527/libs/libs/libopencv_gpu.so /home/mikh/EC527/libs/libs/libopencv_highgui.so /home/mikh/EC527/libs/libs/libopencv_imgproc.so /home/mikh/EC527/libs/libs/libopencv_legacy.so /home/mikh/EC527/libs/libs/libopencv_ml.so /home/mikh/EC527/libs/libs/libopencv_objdetect.so /home/mikh/EC527/libs/libs/libopencv_ocl.so /home/mikh/EC527/libs/libs/libopencv_photo.so /home/mikh/EC527/libs/libs/libopencv_stitching.so /home/mikh/EC527/libs/libs/libopencv_superres.so /home/mikh/EC527/libs/libs/libopencv_ts.so /home/mikh/EC527/libs/libs/libopencv_video.so /home/mikh/EC527/libs/libs/libopencv_videostab.so -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab  

ALL = neural_read_base_case neural_read_single_core neural_read_multi_core_openmp neural_read_multi_core_pthreads

all: $(ALL)

neural_write: main_split_save_vectors.o NeuralNet.o ImageData.o
	$(CC) $(CFLAGS) -o neural_write main_split_save_vectors.o NeuralNet.o ImageData.o $(LDFLAGS)

neural_read_base_case: main_split_load_vectors.o NeuralNet.o
	$(CC) $(CFLAGS) -o neural_read_base_case main_split_load_vectors.o NeuralNet.o -lrt

neural_read_single_core: main_split_load_vectors.o NeuralNet_single_core.o
	$(CC) $(CFLAGS) -o neural_read_single_core main_split_load_vectors.o NeuralNet_single_core.o -lrt

neural_read_multi_core_openmp: main_split_load_vectors.o NeuralNet_multi_core_openmp.o
	$(CC) $(CFLAGS) -o neural_read_multi_core_openmp main_split_load_vectors.o NeuralNet_multi_core_openmp.o -lrt -fopenmp

neural_read_multi_core_pthreads: main_split_load_vectors.o NeuralNet_multi_core_pthreads.o
	$(CC) $(CFLAGS) -pthread -o neural_read_multi_core_pthreads main_split_load_vectors.o NeuralNet_multi_core_pthreads.o -lrt

main_split_save_vectors.o: main_split_save_vectors.cpp
	$(CC) $(CFLAGS) $(INCLUDE) -c main_split_save_vectors.cpp

main_split_load_vectors.o: main_split_load_vectors.cpp
	$(CC) $(CFLAGS) -c main_split_load_vectors.cpp -lrt

NeuralNet.o: Neuron.h Layer.h NeuralNet.cpp NeuralNet.h
	$(CC) $(CFLAGS) -c Neuron.h NeuralNet.cpp

NeuralNet_single_core.o: Neuron.h Layer.h NeuralNet_single_core.cpp NeuralNet.h
	$(CC) $(CFLAGS) -c Neuron.h NeuralNet_single_core.cpp

NeuralNet_multi_core_openmp.o: Neuron.h Layer.h NeuralNet_multi_core_openmp.cpp NeuralNet.h
	$(CC) $(CFLAGS) -c Neuron.h NeuralNet_multi_core_openmp.cpp -fopenmp

NeuralNet_multi_core_pthreads.o: Neuron.h Layer.h NeuralNet_multi_core_pthreads.cpp NeuralNet.h
	$(CC) $(CFLAGS) -pthread -c Neuron.h NeuralNet_multi_core_pthreads.cpp

ImageData.o: ImageData.cpp ImageData.h
	$(CC) $(CFLAGS) $(INCLUDE) -c ImageData.cpp

.PHONY: clean

clean:
	rm -rf *.o *.gch $(ALL)
