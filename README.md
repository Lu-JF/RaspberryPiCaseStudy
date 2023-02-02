# RaspberryPiCaseStudy
The case study of the intenlligent edge computing system based on RaspberryPi and deep learning

## Citation
If the code is used in your research, please cite the article as shown below:
Siliang Lu, Jingfeng Lu, Kang An, Xiaoxian Wang, Qingbo He, Edge Computing on IoT for Machine Signal Processing and Fault Diagnosis: A Review, IEEE Internet of Things Journal, 2023, DOI: 10.1109/JIOT.2023.3239944.

### The environment
The case ideally requires:  
Python>=3.8  
keras>=2.6.0  
scipy>=1.6.2  
numpy>=1.19.5  
PIL>=8.2.0  
Scikit-learn>=0.24.1  
matplotlib>=3.3.4  

### The data set
The data set is a part of the CWRU Bearing Data: https://engineering.case.edu/bearingdatacenter/download-data-file  
All of the data was recorded for motor loads of 0 horsepower, and the motor speeds are 1797 to 1720 RPM.  
There are 7 types vibration signals of the fan end accelerometer (FE) used for this case:  
##
(normal.mat)  the normal viberation signal  
(inner_1.mat) the inner raceway fault with the 0.1778mm diameter  
(inner_2.mat) the inner raceway fault with the 0.3556mm diameter  
(outer_1.mat) the outer raceway fault with the 0.1778mm diameter  
(outer_2.mat) the outer raceway fault with the 0.3556mm diameter  
(ball_1.mat)  the rolling element fault with the 0.1778mm diameter  
(ball_2.mat)  the rolling element fault with the 0.3556mm diameter  
##  

### CreatePngDatasets.py  
The code can transform the raw mat file to the preprocessing png file.  
The method extracts kurtosis of the signal split as the gray value of the image by the interval step.  

### Nets.py
The code includes the component of the model and the buliding method.  
Optional models: RMFFCNN, googLeNet.

### Run.py  
The code will train and test the googLeNet automatically if you have used the 'CreatePngDatasets.py' to bulid the png data sets. Then, it can display the accuracy change with the increasing epoch and draw the confusion matrix according to the test results.

### Run_on_RaspberryPi.py
The code gives an example of how to bulid the intelligent edge computing system based on the deep learning. The 'model.h5' must be existed before you execute the 'Run_on_RaspberryPi.py' for deploying the model on a RaspberryPi, which can be gotten from 'Run.py'.  

It can run with the two different mode for recognizing the fault png on the RaspberryPi. The one is 'Auto', which is scanning the specified path every 2 seconds and ouput the results. The other one is 'Manual', which is triggered scanning by the input command and output the results. Besides, you can output the probability for each type or just output the classification results by the display config. 
