## Table of Contents
  - [What is Docker?](#what-is-docker)
  - [Why you need Docker with this code?](#why-you-need-docker-with-this-code)
  - [The Errors](#the-errors)
  - [Final Solution](#final-solution)
  - [How to build the docker?](#how-to-build-the-docker)




## What is Docker?
Docker is a tool which is used to automate the deployment of applications in lightweight containers so that applications can work efficiently in different environments. You can find the detailed explanation of Docker in the related Xmind file, and also this [video](https://www.youtube.com/watch?v=3c-iBn73dDE) will be helpful. In simple words, its role is the same as the conda environment that I was creating to run the [attention-target-detection demo](https://github.com/ejcgt/attention-target-detection).

## Why you need Docker with this code?
In generall, it is good to create a docker/conda environment in which your code runs perfectly and provide it together with your code. However, in my case it was a necessity to create one. The problem was that the [conda environment](https://github.com/ejcgt/attention-target-detection/blob/master/environment.yml) provided for the source code, utilizes old versions of the most of the packages. To be more clear, it runs the code with the python 3.5.6 and pytorch 0.4.1 which are pretty old versions. This was not a problem running the code alone, however, the problems emerged when I tried to use thid code with YARP. This step is a necessary since it makes the code online and the only way for the code to be able to interact with the iCub is to get wrapped with YARP. After long long try and fails :) the ultimate solution found.

## The Errors
  1. Install YARP separately and run it inside the conda environment
       
      It did not work properly since YARP was using the latest python version and could not idntify the python version specified inside the environment. The attempts to make YARP see the python 3.5 was not successful. In the YARP documentations there is a solution to make YARP use specific version of the python specifying it in the variable `YARP_USE_PYTHON_VERSION`, but we could not find this variable in any of the cmake files.
  2. A docker image with ubuntu 16

    Using a docker with ubuntu16, python3.5 and YARP 3.4 did not work. There was errors while building the docker.


## Final Solution
The final solution was to build a docker starting from ubuntu 18 with python 3.6 which can also see python 3.5, the YARP with latest version, Anaconda and also clone the repository [attention-target-detection demo](https://github.com/ejcgt/attention-target-detection). After building the docker, create the environment related to the .yml file and run the basic demo code. It was successful up to this point.
- Run on CPU
    
  Since nvidia-docker was not installed and the docker environment was not able to see the GPU first I run it only on CPU to test (To do so remove the `.cuda()` whereever you see it in the demo.py code).

## How to build the docker?
