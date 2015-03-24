# Neural Network OCR

We propose the design, implementation, training, and evaluation of a feed-forward neural network adapted to the task of optical character recognition (OCR), in particular as applied to the recognition of handwritten digits. We define a precise metric for measuring the success of such a network on real-world data. We experimentally measure the efficacy of the neural network under variation of certain parameters of its construction and seek to maximize a precise quantity expressing the success of the neural network versus the difficulty in constructing and training it.

## Authors
* Bryan Cuccioli (blc72@cornell.edu)
* Renato Amez (ra374@cornell.edu)

## Training

We define the success of the neural network as the ratio of correctly recognized digits. The network will be trained in handwritten digit-recognition with 10,000 images derived from <a href="http://cis.jhu.edu/~sachin/digit/digit.html">the MNIST database</a> using a <a href="https://gist.github.com/4056614">simple script</a>. We use <a href="http://en.wikipedia.org/wiki/Otsu's_method">Otsu thresholding</a> to convert each image from grayscale to black and white, and resize each image from 28x28 to 6x6, thus reducing the dimensionality of the input. The number of training samples used is a parameter that will be varied for the purpose of experimentation.

## Evaluation

Evaluation of the neural network will be performed using image samples from MNIST distinct from the training data. We will evaluate the system varying parameters such as

* the number of training samples per digit,
* the number of layers in the neural network,
* the number of neurons per hidden layer,
* the weight of the bias,
* the learning rate coefficient used in back-propagation,
* the response threshold for the sigmoid function.

For experimentation purposes, the success of the neural network in performing OCR will be measured as a function of each of these parameters separately. We will experimentally determine the values of these parameters that maximize the training efficiency.

## Building

This program is built using the <a href="http://opencv.willowgarage.com/wiki/">OpenCV</a> library, which is compatible with Windows, OS X, and Linux.

If you are using Linux, you can install OpenCV via most package managers; for example, in Debian/Ubuntu:

    sudo apt-get install libopencv-dev

Or in OSX:

    sudo port install opencv

You may have to change the `INCLUDE` variable in the Makefile to point to the location in which the library is installed.

## Running

Once built, the program may be run with the above parameters set via command-line flags:

* -t: the number of training samples per digit,
* -T: the number of testing samples per digit,
* -l: the number of hidden layers; default = 2,
* -b: the weight of the bias,
* -a: the learning rate for back propagation,
* -r: the response threshold for the sigmoid function,
* -h: the number of neurons per hidden layer.

## License

This project is licensed under the MIT open source license. See the LICENSE
file for details.
