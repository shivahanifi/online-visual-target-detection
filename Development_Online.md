# Developing The Demo Code with YARP
## Table of Contents
  - [Road Map](#road-map)
  - [Data Dump](#data-dump)
  - [Code Modification](#code-modification)
    - [Structure](#structure)
    - [YARP Involved](#yarp-involved)
    - [Object Oriented Programming](#object-oriented-programming)
    - [`configure` function](#configure-function)
    - 
  
  

## Road Map
This step is to complete the final goal, which is to make the whole process online. Since the [offline version](https://github.com/shivahanifi/visual-targets/tree/main/Demo/VT_Demo_Dev_Offline) was successful, here the aim is to involve YARP middleware and run the demo online. 

![Roadmap](Img/roadmap.jpg)

## Data Dump
To frequently test the code, we dumped data using the RealSense camera on the iCub and a `yarpdatadumper`. We had dumped data previously, while [collecting custom input](https://github.com/shivahanifi/visual-targets/tree/main/Demo/VT_Demo_CustomInput/VT_CI_Collection). Here 3 dumpers were used to dump the RGB image, depth and skeleton which is the output of the openpose. Two different sequences of data were collected. You can find them in [VTD_dumper]().

- Note: OpenPose is sensitive and it randomly selects the people in the scene and outputs the skeleton. With the second sequence of data we tried to close the door and avoid problems better.

## Code Modification
### Structure
In oredr to make the code more professional and less confusing, I have followed the structures in [face-recogniser-demo](https://github.com/MariaLombardi/face-recogniser-demo), [leftright-gaze-estimator-demo](https://github.com/MariaLombardi/leftright-gaze-estimator-demo) and [mutual-gaze-classifier-demo](https://github.com/MariaLombardi/mutual-gaze-classifier-demo) repositories.  The constant variables and configuration-related information have been placed in `config_vt.py` file. Additionaly, the functions are defined in the file `utilities_vt.py` and imported to the main demo file when needed. 

The repository contains the `src` folder which includes the main Python code for the demo and the `utilities_vt.py` and `config_vt.py` files. There is also the `app` folder which contains the application XML files for the demo in the  `scripts` and also the initialization files in the `config`. 

- Note: The original code from [attention-target-detection](https://github.com/ejcgt/attention-target-detection) repository has its own `config` and `utils`. However, it is better not to merge them for the sake of debugging and understandability.
  

### YARP Involved
In oredr to make the process online, such that it recieves the RGB images from the RealSense camera and processes it through the code and outputs the attention maps in the real time, YARP needs to be involved. To do so, the first step is to import yarp and call the method `yarp.Network.init()`. This method performs the start-up tasks and includes utilities for manipulating the YARP network, including initialization and shotdown.

Up to this point in the project, in order to run the demo, the environment provided by [attention-target-detection](https://github.com/ejcgt/attention-target-detection) was being used and when YARP needed (i.e. to dump test data, etc) a docker provided by [mutual-gaze-classifier-demo](https://github.com/MariaLombardi/mutual-gaze-classifier-demo/tree/main/app/demo_docker) was involved. However, it is not possible to use them simultaneously (At least as I know:)). I faced several errors when importing YARP and using it together with the environment. 

#### ERRORS
1. DELL laptop has a problem with `SWIG`. it cannot find the `swig4.1.1` and therefor faces errors while installing robotology-superbuild.

  - I tried to install SWIG both in the environment and outside of it. Also tried to change the SWIG direction from cmake but non of them worked. 
  - Also tried to build robotology-superbuild while in the environment using the command:
   ```
   conda install -c conda-forge -c robotology python=3.5
   ``` 
  It didn't work either!

2. With the msi laptop I installed robotology-superbuild successfully on the other environment but [attention-target-detection](https://github.com/ejcgt/attention-target-detection)'s environment was not able to detect it. I was recieving an error related to GlibC

3. Build robotology-superbuild while in the environment
  