To run CPU code, first unzip cpu_code.zip
Then compile code with make all
This will create 4 executable files, base code, single core optimization, OpenMP, and pthread
Each of these can be run in the same way and will produce a results file
The run.sh script included gives examples of how to run the code. 

Files of particular interest are the NeuralNet versions of different optimizations.

To run GPU code, first unzip nvcc_gpu.zip
Then compile code with make.
The key files to look at is the NeuralNet.cu.
To change the optimization, you can change global or shared in the #define and comment/uncomment parts 
of the shared memory implementation for coalescing code. 