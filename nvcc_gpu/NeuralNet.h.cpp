#include "Neuron.h.cpp"
#include "Layer.h.cpp"

class NeuralNet {
private:
  int numInputs;
  int numOutputs;
  int numHiddenLayers;
  int numNeuronsPerLayer;
  float learningRate;
  float responseThreshold;

  std::vector<Layer*>* layers;
  float* outputLayer;

public:
  NeuralNet(int inputs,
            int outputs,
            int hiddenLayers,
            int neuronsPerLayer,
            float alpha,
            float threshold);

  ~NeuralNet();

  float* getWeights() const;

  // Compute the outputs from a given set of inputs.
  void feedForward(std::vector<float>* inputs,
                   std::vector<float>* outputLayer,
                   const float bias);
				   
  // Compute the outputs from a given set of inputs.
  void feedForward_gpu(std::vector<float>* inputs,
                   std::vector<float>* outputLayer,
                   const float bias);
				   
  // Back propagate the errors to update the weights.
  void backPropagate(std::vector<float>* outputs, int teacher);

  // Sigmoid response function.
  inline float sigmoid(float activation);

  // Derivative of sigmoid response function.
  inline float sigmoidPrime(float activation);

  void print() {
    for (int i = 1; i < layers->size(); i++) {
      std::cout << "Layerr #" << i << "\n";
      (*layers)[i]->printNeurons();
    }
  }
};
