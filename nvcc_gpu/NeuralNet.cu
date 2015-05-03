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
#define PRINT_TIMER                    1
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
                            const double bias) 
{
   //GPU Timing variables
   //cudaEvent_t start, stop;
   //float elapsed_gpu;
   
   // Arrays on GPU global memory
   double *d_A;          //Copy of h_A on GPU
   double *d_B;          //Copy of h_B on GPU
   double *d_result;     //MMM result on GPU
   
   // Allocate GPU memory
   size_t allocSize = ARR_LENGTH * ARR_LENGTH * sizeof(double);
   CUDA_SAFE_CALL(cudaMalloc((void **)&d_A, allocSize));
   CUDA_SAFE_CALL(cudaMalloc((void **)&d_B, allocSize));
   CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, allocSize));
   cudaMemset(d_A, 0, allocSize);
   cudaMemset(d_B, 0, allocSize);
   cudaMemset(d_result, 0, allocSize);
   
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
    
	// Free-up device memory
	CUDA_SAFE_CALL(cudaFree(d_A));
	CUDA_SAFE_CALL(cudaFree(d_B));
	CUDA_SAFE_CALL(cudaFree(d_result));
		  
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
