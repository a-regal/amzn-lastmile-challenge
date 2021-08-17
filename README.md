## Amazon Last Mile Challenge Code for Using Graph Neural Networks to Solve Route Sequencing Problems

This repo follows the rc-cli standard of interacting with the dataset for the challenge. Please refer to their installation so you can get the dataset to reproduce this codebase. Within this repo, and specifically the src folder, you will find all code related to the implementation of the graph neural network.

To install all the dependencies, two approaches can be followed:

* Build the docker image
* Use ``` pip install -r requirements.txt```

For the second approach, make sure to check the [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) from pytorch geometric to match the version of torch (or the GPU compilation if needed).
The model build script includes:

* Converting the challenge's data into a directed, bipartite graph structure
* Defining the GNN
* The training loop

The model apply script inclues:

* Loading the network weights into a new instance of the class
* Running the sequencing algorithm (akin to an inverse of the make_dataset function) on each test sequence
* Writing the results in the challenges format 
