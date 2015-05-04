#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

class Neuron {
private:
  int numInputs;
  std::vector<float>* weights;
  float delta;
  float activation;
  float value;

public:
  Neuron() {}

  Neuron(int inputs) {
    numInputs = inputs;
    // There is an extra weight for the bias input.
    weights = new std::vector<float>(numInputs + 1);

    // Setup weights with an initial random value between -1 and 1. There is
    // one weight for each input and an additional bias weight.
    for (int i = 0; i < weights->size(); i++) {
      (*weights)[i] = 10 * (((float)rand() / (float)RAND_MAX) * 2 - 1);
    }
  }

  ~Neuron() {
    delete weights;
  }

  // Get the corresponding weight.
  float getWeight(int n) const {
    return (*weights)[n];
  }

  // Add an update value to a specified input weight.
  void updateWeight(int pos, float update) {
    (*weights)[pos] += update;
  }

  // Get the linear combination of inputs to the neuron.
  float getActivation() const {
    return activation;
  }

  // Get the value of the neuron (sigmoid applied to the activation).
  float getValue() const {
    return value;
  }

  // Set the value of the neuron.
  void setValue(float v) {
    value = v;
  }

  // Get the delta value for this neuron.
  float getDelta() const {
    return delta;
  }

  // Set the delta value for this neuron.
  void setDelta(float new_delta) {
    delta = new_delta;
  }

  // Compute and set the linear combination of inputs to the neuron.
  void setActivation(float a) {
    activation = a;
  }

  void printWeights() {
    for (int i = 0; i < weights->size()-1; i++) {
      std::cout << (*weights)[i] << " ";
    }
    std::cout << "\n";
  }
};
