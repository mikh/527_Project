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

#define GLOBAL                         1
#define SHARED                         0
#define TILE_WIDTH                     16
#define PRINT_TIMER                    1

__global__ void kernel_VMM_global(float* d_A, float* d_B, float* d_result, int UNC) {

  float sum = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  #pragma unroll
  for(int k = 0; k < UNC; k++)
	sum += d_A[k] * d_B[row*UNC+k];  
  d_result[row] = sum; 

}

__global__ void kernel_VMM_shared(float* d_A, float* d_B, float* d_result, int UNC, int CNC)
{
  __shared__ float ds_A[TILE_WIDTH]; 
  int Row = blockIdx.x * TILE_WIDTH + threadIdx.x;
  float sum = 0;

  for (int m = 0; m < (UNC-1)/TILE_WIDTH+1; ++m)
    {
      if((m*TILE_WIDTH + threadIdx.x) < UNC)
      	ds_A[threadIdx.x] = d_A[m*TILE_WIDTH + threadIdx.x]; 
      else
      	ds_A[threadIdx.x] = 0;
		
      __syncthreads();	
      
      for (unsigned int k = 0; k < TILE_WIDTH; k++)
      {	  
          if(Row < CNC && (m*TILE_WIDTH +k) < UNC)
		    //Slow implementation
    		//sum += d_B[(m*TILE_WIDTH) + (Row*UNC)+k] * ds_A[k];
			
			//Faster implementation with coalescing 
			sum += d_B[Row + (k + TILE_WIDTH * m) * CNC] * ds_A[k];
	  }
    	__syncthreads();
    }		
    
  if(Row < CNC)  
    d_result[Row] = sum; 
  __syncthreads();
}


// Initialize the neural network with the given input parameters, in turn
// initializing each layer with neurons of random weight.
NeuralNet::NeuralNet(int inputs,
                     int outputs,
                     int hiddenLayers,
                     int neuronsPerLayer,
                     float alpha,
                     float threshold) {
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
void NeuralNet::feedForward(vector<float>* inputs,
                            vector<float>* outputLayer,
                            const float bias) {
  Layer* inputLayer = (*layers)[0];
  for (int i = 0; i < inputLayer->neuronCount(); i++) {
    inputLayer->getNeuron(i)->setValue((*inputs)[i]);
  }
  for (int l = 1; l < numHiddenLayers + 2; l++) {
    Layer *curr = (*layers)[l], *upstream = (*layers)[l-1];
    for (int j = 0; j < curr->neuronCount(); j++) {
      Neuron *n = curr->getNeuron(j);
      float sum = 0;
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
void NeuralNet::feedForward_gpu(vector<float>* inputs,
                            vector<float>* outputLayer,
                            const float bias) 
{ 
   // Arrays on the host memory
   float *h_up;             //Initial Upstream Vector
   float *h_curr;           //Initial Current Stream Matrix
   float *h_result;         //Copy of MMM result from GPU (d_result)
   
   // Arrays on GPU global memory
   float *d_up;             //Copy of h_A on GPU
   float *d_curr;           //Copy of h_B on GPU
   float *d_result;         //MMM result on GPU   
   
   Layer* inputLayer = (*layers)[0];
  
   size_t allocSizeV = inputLayer->neuronCount() * sizeof(float);
   h_result              = (float *) malloc(allocSizeV);
   memset(h_result, 0, allocSizeV);

   for (int i = 0; i < inputLayer->neuronCount(); i++) {
      inputLayer->getNeuron(i)->setValue((*inputs)[i]);
   }
   for (int l = 1; l < numHiddenLayers + 2; l++) 
   {
      Layer *curr = (*layers)[l], *upstream = (*layers)[l-1];	  
	  int CNC = curr->neuronCount();
	  int UNC = upstream->neuronCount();
	  //printf("CNC = %i, ",CNC);
	  //printf("UNC = %i\n",UNC);
	  
	  //Initialize upstream
	  size_t allocSizeV = UNC * sizeof(float);
	  h_up                = (float *) malloc(allocSizeV);
	  memset(h_up, 0, allocSizeV);

	  //Initialize current stream
	  size_t allocSizeM = CNC * UNC * sizeof(float);
	  h_curr              = (float *) malloc(allocSizeM);
	  memset(h_curr, 0, allocSizeM);  
	  
	  for (int j = 0; j < CNC; j++) 
	  {
         Neuron *n = curr->getNeuron(j);
         for (int i = 0; i < UNC; i++) {
		    h_up[i] = upstream->getNeuron(i)->getValue();
            h_curr[j*UNC+i] = n->getWeight(i);
         }
	  }
	  
      size_t allocSizeC = CNC * sizeof(float);
      // Allocate GPU memory
	  CUDA_SAFE_CALL(cudaMalloc((void **)&d_up, allocSizeV));
	  CUDA_SAFE_CALL(cudaMalloc((void **)&d_curr, allocSizeM));
	  CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, allocSizeC));
	  cudaMemset(d_up, 0, allocSizeV);
	  cudaMemset(d_curr, 0, allocSizeM);
	  cudaMemset(d_result, 0, allocSizeC);
	  
	  // Transfer the arrays to the GPU memory
      CUDA_SAFE_CALL(cudaMemcpy(d_up, h_up, allocSizeV, cudaMemcpyHostToDevice));
	  CUDA_SAFE_CALL(cudaMemcpy(d_curr, h_curr, allocSizeM, cudaMemcpyHostToDevice));

#if GLOBAL      
	  // Launch the kernel
	  int size = UNC/32;
	  dim3 dimGrid(size,size);
	  dim3 dimBlock(32,32);
	  kernel_VMM_global<<<dimGrid, dimBlock>>>(d_up, d_curr, d_result, UNC);
#endif

#if	SHARED
	  int size = UNC/TILE_WIDTH;
	  dim3 dimGrid(size,size);
      dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
	  kernel_VMM_shared<<<dimGrid, dimBlock>>>(d_up, d_curr, d_result, UNC, CNC);
#endif

      // Check for errors during launch
	  CUDA_SAFE_CALL(cudaPeekAtLastError());
	  
	  //Initialize results
	  h_result              = (float *) malloc(allocSizeC);
	  memset(h_result, 0, allocSizeC);  

      // Transfer the results back to the host
      CUDA_SAFE_CALL(cudaMemcpy(h_result, d_result, allocSizeC, cudaMemcpyDeviceToHost));

	  for (int j = 0; j < CNC; j++) 
	  {
         Neuron *n = curr->getNeuron(j);
         n->setActivation(h_result[j]);
         n->setValue(sigmoid(h_result[j]));
      }
/*      
	  for (int j = 0; j < curr->neuronCount(); j++) {
        Neuron *n = curr->getNeuron(j);
        float sum = 0;
        for (int i = 0; i < upstream->neuronCount(); i++) {
          sum += n->getWeight(i) * upstream->getNeuron(i)->getValue();
        }
		//printf("sum = %f, ", sum);
		//printf("hresult = %f\n", h_result[j]);
        n->setActivation(sum);
        n->setValue(sigmoid(sum));
      }
*/

	  // Free-up memory
	  CUDA_SAFE_CALL(cudaFree(d_up));
	  CUDA_SAFE_CALL(cudaFree(d_curr));
	  CUDA_SAFE_CALL(cudaFree(d_result));
	
      // Free-up memory
  	  free(h_up);
      free(h_curr);
	  free(h_result);
    }

    Layer* lastLayer = (*layers)[numHiddenLayers+1];
    for (int i = 0; i < lastLayer->neuronCount(); i++) 
	{
       (*outputLayer)[i] = lastLayer->getNeuron(i)->getValue();
    }
}

// Back propagate the errors to update the weights.
void NeuralNet::backPropagate(vector<float>* outputs, int teacher) {
  Layer *outputLayer = (*layers)[numHiddenLayers + 1];
  for (int i = 0; i < outputs->size(); i++) {
    Neuron *n = outputLayer->getNeuron(i);
    float adjusted = -n->getValue();
    if (i == teacher) {
      adjusted += 1;
    }
    n->setDelta(sigmoidPrime(n->getActivation()) * adjusted);
  }

  // Propagate deltas backward from output layer to input layer.
  for (int l = numHiddenLayers; l >= 0; l--) {
    Layer *curr = (*layers)[l], *downstream = (*layers)[l+1];

    for (int i = 0; i < curr->neuronCount(); i++) {
      float sum = 0;
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

// Back propagate the errors to update the weights.
void NeuralNet::backPropagate_gpu(vector<float>* outputs, int teacher) {
   Layer *outputLayer = (*layers)[numHiddenLayers + 1];
   for (int i = 0; i < outputs->size(); i++) 
   {
      Neuron *n = outputLayer->getNeuron(i);
      float adjusted = -n->getValue();
      if (i == teacher) 
	  {
         adjusted += 1;
      }
      n->setDelta(sigmoidPrime(n->getActivation()) * adjusted);
   }

   // Arrays on the host memory
   float *h_weight;         //Initial Upstream Vector
   float *h_delta;          //Initial Current Stream Matrix
   float *h_result;         //Copy of MMM result from GPU (d_result)
   
   // Arrays on GPU global memory
   float *d_weight;         //Copy of h_A on GPU
   float *d_delta;          //Copy of h_B on GPU
   float *d_result;         //MMM result on GPU   
   
   // Propagate deltas backward from output layer to input layer.
   for (int l = numHiddenLayers; l >= 0; l--) 
   {
      Layer *curr = (*layers)[l], *downstream = (*layers)[l+1];
	  int CNC = curr->neuronCount();
	  int DNC = downstream->neuronCount();
	  //printf("CNC = %i, ",CNC);
	  //printf("DNC = %i\n",DNC);
	
	  //Initialize weight
	  size_t allocSizeM = DNC * CNC * sizeof(float);
	  h_weight            = (float *) malloc(allocSizeM);
	  memset(h_weight, 0, allocSizeM);

	  //Initialize delta
	  size_t allocSizeV = DNC * sizeof(float);
	  h_delta              = (float *) malloc(allocSizeV);
	  memset(h_delta, 0, allocSizeV);  
	  
      for (int i = 0; i < CNC; i++) 
	  {
         for (int j = 0; j < DNC; j++) 
	     {
           h_weight[i*DNC+j] = downstream->getNeuron(j)->getWeight(i);
		   h_delta[j] = downstream->getNeuron(j)->getDelta();
         }
      }

      size_t allocSizeC = CNC * sizeof(float);
      // Allocate GPU memory
	  CUDA_SAFE_CALL(cudaMalloc((void **)&d_weight, allocSizeM));
	  CUDA_SAFE_CALL(cudaMalloc((void **)&d_delta, allocSizeV));
	  CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, allocSizeC));
	  cudaMemset(d_weight, 0, allocSizeM);
	  cudaMemset(d_delta, 0, allocSizeV);
	  cudaMemset(d_result, 0, allocSizeC);
	  
	  // Transfer the arrays to the GPU memory
      CUDA_SAFE_CALL(cudaMemcpy(d_weight, h_weight, allocSizeM, cudaMemcpyHostToDevice));
	  CUDA_SAFE_CALL(cudaMemcpy(d_delta, h_delta, allocSizeV, cudaMemcpyHostToDevice));
      
	  // Launch the kernel
#if GLOBAL  
	  int size = CNC/32 +0.5;
	  dim3 dimGrid(size,size);
      dim3 dimBlock(32,32);
      kernel_VMM_global<<<dimGrid, dimBlock>>>(d_delta, d_weight, d_result, DNC);
#endif
	  
#if SHARED	  
	  int size = CNC/TILE_WIDTH;
	  dim3 dimGrid(size,size);
      dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
	  kernel_VMM_shared<<<dimGrid, dimBlock>>>(d_delta, d_weight, d_result, DNC, CNC);
#endif
      // Check for errors during launch
	  CUDA_SAFE_CALL(cudaPeekAtLastError());
	  
	  //Initialize results
	  h_result              = (float *) malloc(allocSizeC);
	  memset(h_result, 0, allocSizeC);  

      // Transfer the results back to the host
      CUDA_SAFE_CALL(cudaMemcpy(h_result, d_result, allocSizeC, cudaMemcpyDeviceToHost));
	
      for (int i = 0; i < CNC; i++) 
	  {
         Neuron *n = curr->getNeuron(i);
	/*
	float sum = 0;
      for (int j = 0; j < downstream->neuronCount(); j++) {
        sum += downstream->getNeuron(j)->getWeight(i)
            * downstream->getNeuron(j)->getDelta();
      }
	  printf("sum = %f, ", sum);
	  printf("hresult = %f\n", h_result[i]);
	*/ 
		 n->setDelta(sigmoidPrime(n->getActivation()) * h_result[i]);
		 
         for (int j = 0; j < DNC; j++) 
	     {
           downstream->getNeuron(j)->updateWeight(i,
		      learningRate * 
			  sigmoid(n->getActivation()) * 
			  downstream->getNeuron(j)->getDelta());
         }
      }
	  
	// Free-up memory
	CUDA_SAFE_CALL(cudaFree(d_weight));
	CUDA_SAFE_CALL(cudaFree(d_delta));
	CUDA_SAFE_CALL(cudaFree(d_result));
	
    // Free-up memory
  	free(h_weight);
    free(h_delta);
	free(h_result);
   }
}

// Compute the sigmoid function.
inline float NeuralNet::sigmoid(float activation) {
  return 1.0 / (1.0 + exp(-activation / responseThreshold));
}

// Compute the derivative of the sigmoid function
inline float NeuralNet::sigmoidPrime(float activation) {
  float exponential = exp(activation / responseThreshold);
  return exponential / (responseThreshold * pow(exponential + 1, 2));
}
