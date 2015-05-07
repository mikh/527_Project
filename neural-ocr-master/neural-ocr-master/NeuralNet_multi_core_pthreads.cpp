#include <cmath>
#include <cstdio>
#include <iostream>
#include <pthread.h>

#include "NeuralNet_pthread.h"

using namespace std;

#define LOOP_UNROLLING 12
#define LOOP_UNROLLING_BACK_PROP 12

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

void NeuralNet::double_loop_work_thread_ff(Layer *curr, Layer *upstream, int N, int K){
  for (int j = 0; j < N; j++) {
      Neuron *n = curr->getNeuron(j);
      double sum = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0, sum9 = 0, sum10 = 0, sum11 = 0, sum12 = 0, sum13 = 0;
      int i = 0;
      for (i = LOOP_UNROLLING; i < K; i+=(LOOP_UNROLLING+1)) {
        sum1 += n->getWeight(i) * upstream->getNeuron(i)->getValue();
        sum2 += n->getWeight(i-1) * upstream->getNeuron(i-1)->getValue();
        sum3 += n->getWeight(i-2) * upstream->getNeuron(i-2)->getValue();
        sum4 += n->getWeight(i-3) * upstream->getNeuron(i-3)->getValue();
        sum5 += n->getWeight(i-4) * upstream->getNeuron(i-4)->getValue();
        sum6 += n->getWeight(i-5) * upstream->getNeuron(i-5)->getValue();
        sum7 += n->getWeight(i-6) * upstream->getNeuron(i-6)->getValue();
        sum8 += n->getWeight(i-7) * upstream->getNeuron(i-7)->getValue();
        sum9 += n->getWeight(i-8) * upstream->getNeuron(i-8)->getValue();
        sum10 += n->getWeight(i-9) * upstream->getNeuron(i-9)->getValue();
        sum11 += n->getWeight(i-10) * upstream->getNeuron(i-10)->getValue();
        sum12 += n->getWeight(i-11) * upstream->getNeuron(i-11)->getValue();
        sum13 += n->getWeight(i-12) * upstream->getNeuron(i-12)->getValue();
      }
      sum = sum1+sum2+sum3+sum4+sum5 + sum6 + sum7+sum8+sum9+sum10+sum11+sum12+sum13;
      for (i; i < K; i++) {
        sum += n->getWeight(i) * upstream->getNeuron(i)->getValue();
      }
      n->setActivation(sum);
      n->setValue(sigmoid(sum));
    }
}

// Compute the sigmoid function.
inline double sigmoid(double activation, double responseThreshold) {
  return 1.0 / (1.0 + exp(-activation / responseThreshold));
}

  struct thread_parameters{
    Layer *upstream;
    int K;
    Neuron *n;
    double responseThreshold;
  };

void* single_loop_work_thread_ff(void *ptr){
      struct thread_parameters *p = (struct thread_parameters*) ptr;
      Layer *upstream = p->upstream;
      int K = p->K;
      Neuron *n = p->n;
      double responseThreshold = p->responseThreshold;

     double sum = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0, sum9 = 0, sum10 = 0, sum11 = 0, sum12 = 0, sum13 = 0;
      int i = 0;
      for (i = LOOP_UNROLLING; i < K; i+=(LOOP_UNROLLING+1)) {
        sum1 += n->getWeight(i) * upstream->getNeuron(i)->getValue();
        sum2 += n->getWeight(i-1) * upstream->getNeuron(i-1)->getValue();
        sum3 += n->getWeight(i-2) * upstream->getNeuron(i-2)->getValue();
        sum4 += n->getWeight(i-3) * upstream->getNeuron(i-3)->getValue();
        sum5 += n->getWeight(i-4) * upstream->getNeuron(i-4)->getValue();
        sum6 += n->getWeight(i-5) * upstream->getNeuron(i-5)->getValue();
        sum7 += n->getWeight(i-6) * upstream->getNeuron(i-6)->getValue();
        sum8 += n->getWeight(i-7) * upstream->getNeuron(i-7)->getValue();
        sum9 += n->getWeight(i-8) * upstream->getNeuron(i-8)->getValue();
        sum10 += n->getWeight(i-9) * upstream->getNeuron(i-9)->getValue();
        sum11 += n->getWeight(i-10) * upstream->getNeuron(i-10)->getValue();
        sum12 += n->getWeight(i-11) * upstream->getNeuron(i-11)->getValue();
        sum13 += n->getWeight(i-12) * upstream->getNeuron(i-12)->getValue();
      }
      sum = sum1+sum2+sum3+sum4+sum5 + sum6 + sum7+sum8+sum9+sum10+sum11+sum12+sum13;
      for (i; i < K; i++) {
        sum += n->getWeight(i) * upstream->getNeuron(i)->getValue();
      }
      n->setActivation(sum);
      n->setValue(sigmoid(sum, responseThreshold));
      
  }


// Compute the outputs from a given set of inputs.
void NeuralNet::feedForward(vector<double>* inputs,
                            vector<double>* outputLayer,
                            const double bias) {
  Layer* inputLayer = (*layers)[0];
  vector<pthread_t> threads;
  int N = inputLayer->neuronCount(), K;
  for (int i = 0; i < N; i++) {
    inputLayer->getNeuron(i)->setValue((*inputs)[i]);
  }
  for (int l = 1; l < numHiddenLayers + 2; l++) {
    Layer *curr = (*layers)[l], *upstream = (*layers)[l-1];
    N = curr->neuronCount();
    K = upstream->neuronCount();
    threads.clear();
    for (int j = 0; j < N; j++) {
     // printf("threading %d\n", N);
      Neuron *n = curr->getNeuron(j);
      thread_parameters p;
      p.upstream = upstream;
      p.K = K;
      p.n = n;
      p.responseThreshold = responseThreshold;
      pthread_t new_pthread;
      pthread_create(&new_pthread, NULL, single_loop_work_thread_ff, &p);
      threads.push_back(new_pthread);
      //single_loop_work_thread_ff(upstream, K, n);
    }
    for(int j = 0; j < threads.size(); j++){
  //    printf("joining");
      pthread_join(threads[j], NULL);
    }

  }

  Layer* lastLayer = (*layers)[numHiddenLayers+1];
  for (int i = 0; i < lastLayer->neuronCount(); i++) {
    (*outputLayer)[i] = lastLayer->getNeuron(i)->getValue();
  }
  printf("Feed forward end\n");
}

// Back propagate the errors to update the weights.
void NeuralNet::backPropagate(vector<double>* outputs, int teacher) {
  Layer *outputLayer = (*layers)[numHiddenLayers + 1];
  int N = outputs->size(), K;
  for (int i = 0; i < N; i++) {
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
    K = downstream->neuronCount();


    N = curr->neuronCount();
    for (int i = 0; i < N; i++) {
      double sum = 0, sum1=0, sum2=0, sum3=0, sum4=0, sum5=0, sum6=0, sum7=0, sum8=0, sum9=0, sum10=0, sum11=0, sum12=0;
      Neuron *n = curr->getNeuron(i);
      int j;
      for (j = LOOP_UNROLLING_BACK_PROP; j < K; j+=(LOOP_UNROLLING_BACK_PROP+1)) {
        sum1 += downstream->getNeuron(j)->getWeight(i) * downstream->getNeuron(j)->getDelta();
        sum2 += downstream->getNeuron(j-1)->getWeight(i) * downstream->getNeuron(j-1)->getDelta();
        sum3 += downstream->getNeuron(j-2)->getWeight(i) * downstream->getNeuron(j-2)->getDelta();
        sum4 += downstream->getNeuron(j-3)->getWeight(i) * downstream->getNeuron(j-3)->getDelta();
        sum5 += downstream->getNeuron(j-4)->getWeight(i) * downstream->getNeuron(j-4)->getDelta();
        sum6 += downstream->getNeuron(j-5)->getWeight(i) * downstream->getNeuron(j-5)->getDelta();
        sum7 += downstream->getNeuron(j-6)->getWeight(i) * downstream->getNeuron(j-6)->getDelta();
        sum8 += downstream->getNeuron(j-7)->getWeight(i) * downstream->getNeuron(j-7)->getDelta();
        sum9 += downstream->getNeuron(j-8)->getWeight(i) * downstream->getNeuron(j-8)->getDelta();
        sum10 += downstream->getNeuron(j-9)->getWeight(i) * downstream->getNeuron(j-9)->getDelta();
        sum11 += downstream->getNeuron(j-10)->getWeight(i) * downstream->getNeuron(j-10)->getDelta();
        sum12 += downstream->getNeuron(j-11)->getWeight(i) * downstream->getNeuron(j-11)->getDelta();
      }
      sum = sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9+sum10+sum11+sum12;
      for (j; j < K; j++) {
        sum += downstream->getNeuron(j)->getWeight(i) * downstream->getNeuron(j)->getDelta();
      }
      n->setDelta(sigmoidPrime(n->getActivation()) * sum);
     /* for (j = LOOP_UNROLLING_BACK_PROP; j < K; j+=(LOOP_UNROLLING_BACK_PROP+1)) {
        downstream->getNeuron(j)->updateWeight(i, learningRate * sigmoid(n->getActivation()) * downstream->getNeuron(j)->getDelta());
        downstream->getNeuron(j-1)->updateWeight(i, learningRate * sigmoid(n->getActivation()) * downstream->getNeuron(j-1)->getDelta());
        downstream->getNeuron(j-2)->updateWeight(i, learningRate * sigmoid(n->getActivation()) * downstream->getNeuron(j-2)->getDelta());
        downstream->getNeuron(j-3)->updateWeight(i, learningRate * sigmoid(n->getActivation()) * downstream->getNeuron(j-3)->getDelta());
        downstream->getNeuron(j-4)->updateWeight(i, learningRate * sigmoid(n->getActivation()) * downstream->getNeuron(j-4)->getDelta());
        downstream->getNeuron(j-5)->updateWeight(i, learningRate * sigmoid(n->getActivation()) * downstream->getNeuron(j-5)->getDelta());
      }*/
      j=0;
      for (j; j < K; j++) {
        downstream->getNeuron(j)->updateWeight(i, learningRate * sigmoid(n->getActivation()) * downstream->getNeuron(j)->getDelta());
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
