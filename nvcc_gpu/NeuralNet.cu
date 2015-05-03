#include <cmath>
#include <cstdio>
#include <iostream>

#include "NeuralNet.h.cpp"

using namespace std;
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

#define TILE_WIDTH                     16
#define NUM_BLOCKS                     ARR_LENGTH/TILE_WIDTH
#define PRINT_TIMER                    0
#define ARR_LENGTH                     64

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
// Initialize the neural network with the given input parameters, in turn
// initializing each layer with neurons of random weight.
NeuralNet::NeuralNet(int inputs,
                     int outputs,
                     int hiddenLayers,
                     int neuronsPerLayer,
                     double alpha,
                     double threshold) {
  numInputs = inputs;
  numOutputs = outputs;
  numHiddenLayers = hiddenLayers;
  numNeuronsPerLayer = neuronsPerLayer;
  learningRate = alpha;
  responseThreshold = threshold;
  layers = new vector<Layer*>(hiddenLayers + 2);

  // Initialize each hidden layer.
  (*layers)[0] = new Layer(inputs, 0);
  (*layers)[1] = new Layer(neuronsPerLayer, inputs);
  (*layers)[hiddenLayers + 1] = new Layer(outputs, neuronsPerLayer);
  for (int i = 2; i < layers->size() - 1; i++) {
    (*layers)[i] = new Layer(neuronsPerLayer, neuronsPerLayer);
  }
}

NeuralNet::~NeuralNet() {
  for (int i = 0; i < layers->size(); i++) {
    delete (*layers)[i];
  }
  delete layers;
}

// Compute the outputs from a given set of inputs.
void NeuralNet::feedForward(vector<double>* inputs,
                            vector<double>* outputLayer,
                            const double bias) {
  Layer* inputLayer = (*layers)[0];
  for (int i = 0; i < inputLayer->neuronCount(); i++) {
    inputLayer->getNeuron(i)->setValue((*inputs)[i]);
  }
  for (int l = 1; l < numHiddenLayers + 2; l++) {
    Layer *curr = (*layers)[l], *upstream = (*layers)[l-1];
    for (int j = 0; j < curr->neuronCount(); j++) {
      Neuron *n = curr->getNeuron(j);
      double sum = 0;
      for (int i = 0; i < upstream->neuronCount(); i++) {
        sum += n->getWeight(i) * upstream->getNeuron(i)->getValue();
      }
      n->setActivation(sum);
      n->setValue(sigmoid(sum));
    }
  }

  Layer* lastLayer = (*layers)[numHiddenLayers+1];
  for (int i = 0; i < lastLayer->neuronCount(); i++) {
    (*outputLayer)[i] = lastLayer->getNeuron(i)->getValue();
  }
}

// Compute the outputs from a given set of inputs.
void NeuralNet::feedForward_gpu(vector<double>* inputs,
                            vector<double>* outputLayer,
                            const double bias) 
{
#if PRINT_TIMER
   //GPU Timing variables
   cudaEvent_t start, stop;
   float elapsed_gpu;
#endif
  
   // Arrays on the host memory
   float *h_up;             //Initial Matrix x
   float *h_curr;           //Initial Matrix y
   //float *h_result;         //Copy of MMM result from GPU (d_result)
   
   // Arrays on GPU global memory
   float *d_up;          //Copy of h_A on GPU
   float *d_curr;        //Copy of h_B on GPU
   //float *d_result;      //MMM result on GPU   
   
   Layer* inputLayer = (*layers)[0];
  
   size_t allocSize = inputLayer->neuronCount() * sizeof(float);
   h_curr              = (float *) malloc(allocSize);
   memset(h_curr, 0, allocSize);

   for (int i = 0; i < inputLayer->neuronCount(); i++) {
      inputLayer->getNeuron(i)->setValue((*inputs)[i]);
	  h_curr[i] = (float) (inputLayer->getNeuron(i)->getValue());
   }
   for (int l = 1; l < numHiddenLayers + 2; l++) {
      Layer *curr = (*layers)[l], *upstream = (*layers)[l-1];
	  //#########################################################
	  //printf("curr neuron = %d  ", curr->neuronCount());  
	  size_t allocSize = curr->neuronCount() * sizeof(float);
	  
	  //Initialize upstream
	  h_up                = (float *) malloc(allocSize);
	  memset(h_up, 0, allocSize);
	  for (int m = 0; m < curr->neuronCount(); m++){
	     h_up[m] = h_curr[m];
	  }
	  
	  //Initialize current stream
	  h_curr              = (float *) malloc(allocSize);
	  memset(h_curr, 0, allocSize);  
	  for (int j = 0; j < curr->neuronCount(); j++) 
	  {
	     h_curr[j] = curr->getNeuron(j)->getValue();
      }
      //printf("hello4\n");
      // Transfer the arrays to the GPU memory
	  CUDA_SAFE_CALL(cudaMalloc((void **)&d_up, allocSize));
	  CUDA_SAFE_CALL(cudaMalloc((void **)&d_curr, allocSize));
	  cudaMemset(d_up, 0, allocSize);
	  cudaMemset(d_curr, 0, allocSize);
	  
      CUDA_SAFE_CALL(cudaMemcpy(d_up, h_up, allocSize, cudaMemcpyHostToDevice));
	  CUDA_SAFE_CALL(cudaMemcpy(d_curr, h_curr, allocSize, cudaMemcpyHostToDevice));
      //printf("hello5\n");
	  //##########################################################
      for (int j = 0; j < curr->neuronCount(); j++) 
	  {
         Neuron *n = curr->getNeuron(j);
         double sum = 0;
         for (int i = 0; i < upstream->neuronCount(); i++) {
            sum += n->getWeight(i) * upstream->getNeuron(i)->getValue();
         }
       n->setActivation(sum);
       n->setValue(sigmoid(sum));
       }
    }

    Layer* lastLayer = (*layers)[numHiddenLayers+1];
    for (int i = 0; i < lastLayer->neuronCount(); i++) 
	{
       (*outputLayer)[i] = lastLayer->getNeuron(i)->getValue();
    }
  
#if PRINT_TIMER
	// Create the cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	// Record event on the default stream
	cudaEventRecord(start, 0);
#endif

	// Launch the kernel
	//dim3 dimGrid(NUM_BLOCKS,NUM_BLOCKS);
    //dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
	//kernel_MMM_shared<<<dimGrid, dimBlock>>>(d_up, d_curr, d_result);

#if PRINT_TIMER
	// Stop and destroy the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_gpu,start, stop);
	printf("\nGPU time: %f (msec)\n", elapsed_gpu);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif
    // Check for errors during launch
	//CUDA_SAFE_CALL(cudaPeekAtLastError());
	
    // Transfer the results back to the host
    //CUDA_SAFE_CALL(cudaMemcpy(h_result, d_result, allocSize, cudaMemcpyDeviceToHost));
	
    // Free-up memory
	CUDA_SAFE_CALL(cudaFree(d_up));
	CUDA_SAFE_CALL(cudaFree(d_curr));
	
    // Free-up memory
  	free(h_up);
    free(h_curr);
}

// Back propagate the errors to update the weights.
void NeuralNet::backPropagate(vector<double>* outputs, int teacher) {
  Layer *outputLayer = (*layers)[numHiddenLayers + 1];
  for (int i = 0; i < outputs->size(); i++) {
    Neuron *n = outputLayer->getNeuron(i);
    double adjusted = -n->getValue();
    if (i == teacher) {
      adjusted += 1;
    }
    n->setDelta(sigmoidPrime(n->getActivation()) * adjusted);
  }

  // Propagate deltas backward from output layer to input layer.
  for (int l = numHiddenLayers; l >= 0; l--) {
    Layer *curr = (*layers)[l], *downstream = (*layers)[l+1];

    for (int i = 0; i < curr->neuronCount(); i++) {
      double sum = 0;
      Neuron *n = curr->getNeuron(i);
      for (int j = 0; j < downstream->neuronCount(); j++) {
        sum += downstream->getNeuron(j)->getWeight(i)
            * downstream->getNeuron(j)->getDelta();
      }
      n->setDelta(sigmoidPrime(n->getActivation()) * sum);
      for (int j = 0; j < downstream->neuronCount(); j++) {
        downstream->getNeuron(j)->updateWeight(i,
            learningRate * sigmoid(n->getActivation())
            * downstream->getNeuron(j)->getDelta());
      }
    }
  }
}

// Compute the sigmoid function.
inline double NeuralNet::sigmoid(double activation) {
  return 1.0 / (1.0 + exp(-activation / responseThreshold));
}

// Compute the derivative of the sigmoid function
inline double NeuralNet::sigmoidPrime(double activation) {
  double exponential = exp(activation / responseThreshold);
  return exponential / (responseThreshold * pow(exponential + 1, 2));
}
