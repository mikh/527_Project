#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <cstring>
#include <time.h>
#include <math.h>

//#include "ImageData.h"
#include "NeuralNet.h"

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


#define IMG_SIZE 6*6
#define ALPHABET_SIZE 10

#define GIG 1000000000
#define NANO_TO_MILLI 1000000
#define CPG 2.53         // Cycles per GHz -- Adjust to your computer

#define TILE_WIDTH                     16
#define NUM_BLOCKS                     ARR_LENGTH/TILE_WIDTH
#define PRINT_TIMER                    1
#define TOL                            4e-7
#define ARR_LENGTH                     2048

using namespace std;

void save_double_results(vector<double>* data, char* data_location);
void load_double_results(vector<double>* data, char* data_location);

__global__ void kernel_MMM_shared(float* d_A, float* d_B, float* d_result) {
  __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
  
  float sum = 0;
  long int m,k;
  
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  for(m = 0; m < ARR_LENGTH/TILE_WIDTH; ++m)
  {
   ds_A[threadIdx.y][threadIdx.x] = d_A[row*ARR_LENGTH + (m*TILE_WIDTH + threadIdx.x)];
    ds_B[threadIdx.y][threadIdx.x] = d_B[col + (m*TILE_WIDTH + threadIdx.y)*ARR_LENGTH];
   __syncthreads();

    for (k = 0; k < TILE_WIDTH; ++k)
      sum += ds_A[threadIdx.y][k] * ds_B[k][threadIdx.x];
    __syncthreads();
  }
  
  d_result[row*ARR_LENGTH+col] = sum;
}


int process_ocr(bool training, NeuralNet& nn, double bias, int iterations) {
  struct timespec diff(struct timespec start, struct timespec end);
  struct timespec time1, time2, elapsed_cpu;

  // GPU Timing variables
   cudaEvent_t start, stop;
   float elapsed_gpu;
   
   // Arrays on GPU global memory
   vector<double> *d_A;          //Copy of h_A on GPU
   vector<double> *d_B;          //Copy of h_B on GPU
   vector<double> *d_result;     //MMM result on GPU
   
   // Arrays on the host memory
   vector<double> *h_A;             //Initial Matrix x
   vector<double> *h_B;             //Initial Matrix y
   vector<double> *h_result_gold;   //MMM result on CPU
   vector<double> *h_result;        //Copy of MMM result from GPU (d_result)
   
   int i, j, errCount = 0;
   printf("Size of the Matrix is = %d by %d\n", ARR_LENGTH,ARR_LENGTH);
   
   // Allocate GPU memory
   size_t allocSize = ARR_LENGTH * ARR_LENGTH * sizeof(vector<double>);
   CUDA_SAFE_CALL(cudaMalloc((void **)&d_A, allocSize));
   CUDA_SAFE_CALL(cudaMalloc((void **)&d_B, allocSize));
   CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, allocSize));
   cudaMemset(d_A, 0, allocSize);
   cudaMemset(d_B, 0, allocSize);
   cudaMemset(d_result, 0, allocSize);
   
   // Allocate arrays on host memory
   h_A                = (vector<double> *) malloc(allocSize);
   h_B                = (vector<double> *) malloc(allocSize);
   h_result           = (vector<double> *) malloc(allocSize);
   h_result_gold      = (vector<double> *) malloc(allocSize);
   memset(h_A, 0, allocSize);
   memset(h_B, 0, allocSize);
   memset(h_result, 0, allocSize);
   memset(h_result_gold, 0, allocSize);
   
  int correct = 0;
  int target_size = 6;
  char file_string[100];

  vector<double>* inputs = new vector<double>(IMG_SIZE);
  vector<double>* outputs = new vector<double>(ALPHABET_SIZE);

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);


  for (int j = 0; j < iterations; j++) {
    for (int i = 0; i < ALPHABET_SIZE; i++) {
      delete outputs;

      /*
      ostringstream os;
      os << "data/" << i << "/data" << i << "_" << j << ".jpg";
      ImageData input(os.str(), target_size, false);
      if (input.error()&&0) {
        cout << "Error reading " << os.str() << "\n";
        delete inputs;
        return 1;
      }
      input.getPixels(inputs);
      */

      sprintf(file_string, "data_text/%d/data_%d_%d.txt", i, i, j);
   //   save_double_results(inputs, string(file_string));

      
      load_double_results(inputs, file_string);

      outputs = new vector<double>(ALPHABET_SIZE);
      nn.feedForward(inputs, outputs, bias);

      if (training) {
        double max_val = 0;
        int max_index = 0;
        for (int k = 0; k < outputs->size(); k++) {
          if ((*outputs)[k] > max_val) {
            max_val = (*outputs)[k];
            max_index = k;
          }
        }
        if (max_index == i) {
          correct++;
        }
      } else {
        nn.backPropagate(outputs, i);
      }
      
    }
  }

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
  elapsed_cpu = diff(time1, time2);

  printf("\nCPU time: %f(msec)\n", (float)(((double)GIG*elapsed_cpu.tv_sec + elapsed_cpu.tv_nsec)/(double)NANO_TO_MILLI));


  delete inputs;
  delete outputs;
  return correct;
}

int main(int argc, char *argv[]) {
  srand((unsigned)time(NULL));

  int training = 0, layers = 2, testing = 0;
  double bias = 0, responseThreshold = 1, learningRate = 1;
  int layerHeight = 10;

  // argc is 1 if the command line was given the name of the binary
  // and no additional parameters.
  if (argc == 1) {
    cout << "usage: " << argv[0] << " -t # -l # -b # -a # -r # -h #\n"
         << "-t: the number of training samples per digit.\n"
         << "-T: the number of testing samples per digit.\n"
         << "-l: the number of hidden layers; default = 2.\n"
         << "-b: the weight of the bias.\n"
         << "-a: the learning rate for back propagation.\n"
         << "-r: the response threshold for the sigmoid function.\n"
         << "-h: the number of neurons per hidden layer.\n";
    return 0;
  }

  // Process command line arguments.
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-t") == 0) {
      training = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-T") == 0) {
      testing = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-l") == 0) {
      layers = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-b") == 0) {
      bias = atof(argv[++i]);
    } else if (strcmp(argv[i], "-r") == 0) {
      responseThreshold = atof(argv[++i]);
    } else if (strcmp(argv[i], "-a") == 0) {
      learningRate = atof(argv[++i]);
    } else if (strcmp(argv[i], "-h") == 0) {
      layerHeight = atoi(argv[++i]);
    }
  }

  if (layers < 0 || training <= 0 || testing <= 0 || responseThreshold <= 0
      || layerHeight <= 0 || learningRate < 0) {
    cout << "Invalid argument specified.\n";
    return 1;
  }

  NeuralNet nn(IMG_SIZE,
               ALPHABET_SIZE,
               layers,
               layerHeight,
               learningRate,
               responseThreshold);

  process_ocr(false, nn, bias, training);
  int correct = process_ocr(true, nn, bias, testing);

  cout << "Success: " << correct << " / " << testing * 10
       << " (" << ((double)correct / (double)testing * 10) << "%)\n";

  return 0;
}


void save_double_results(vector<double>* data, char* data_location){
  FILE *pFile;
  pFile = fopen(data_location, "w");
    for(int ii = 0; ii < data->size(); ii++){
      fprintf(pFile, "%.15f\n", (*data)[ii]);
    }
  fclose(pFile);
}

void load_double_results(vector<double>* data, char* data_location){
  data->clear();
  ifstream file;
  file.open(data_location);
  string str;
  while(getline(file, str)){
    data->push_back(strtod(str.c_str(), NULL));
  }
  file.close();
}

  

 


struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}