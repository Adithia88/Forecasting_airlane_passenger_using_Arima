# Time series analysis with ARIMA
Simple python example on how to use ARIMA models to analyze and predict time series

# Forecasting airline passengers

This is an implementation of Time series analysis with arima on Python 3, Keras, and TensorFlow. The model can forecast how many passengers in th future.

![Time series analysis sample](assets/1.PNG)

The repository includes:
* Airline passengers Data
* Training code 
* Testing code
* Jupyter notebooks to visualize the detection pipeline at every step
* Evaluation code


# Getting Started
* open sample Load_model R50_E08.ipynb in the main page  Is the easiest way to start. It shows an example of plant instance segmentation (testing only)
* convert json to COCO
* open train_tunning.ipynb in the main page, this codes for training (training)
* open Evaluation_Gen_IoU.ipynb, test_mask_rcnn_map.ipynb and test_mask_rcnn_dice_coef_test_data.py for evaluation (evaluation)


# step to convert json to COCO 

## 1. Prepare dataset and make sure the path 
open code labelme2coco.py in dataset/custom  and  put dataset (dataset/custom) too in your main path. Make sure u put the data and codes like my picture below!
like u see i divide my data to 3 part train, val and test (just ignore the .json that the result when u convert it)

![](assets/3.PNG)

## 2.convert json to COCO 
just change the name folder what u want convert , so in this case we convert 3 times (train,val and test) 

![](assets/4.PNG)

## 3. Execute program (Labelme2coco.py) 
press F5.


# Step to train  with your own data

## 1. train own dataset
This example will explain which part u must change to train your own dataset. open code with name train_tuning.ipynb  or train_tuning.py in sample folder (Ipynb for jupyter , Py for spyder)  in this tutorial im gonna use jupyter notebook (.ipynb)

the line codes u must change 

![](assets/5.PNG)

Root_dir = (path) which u put whole files 

![](assets/6.PNG)

NUM_classes = 1 + (your own data class)* 
BACKBONE = 'resnet50'  (u can use 2 backbone here resnet50 or resnet101)

![](assets/7.PNG)

dataset_train.load_data(Root_dir + your data train path )
dataset_val.load_data(Root_dir + your data val path )

![](assets/8.PNG)

epoch =  u can config the epoch i make it 25 for heads

![](assets/9.PNG)

epoch =  2x heads epochs (!warning just follow what i said)

![](assets/10.PNG)

so this is bonus stage u can load directly yours model in 1 code  
real_test_dir = Root_dir + (your test data path) 

## 3. Execute all program
press shift+enter per line or Run all 



# Step to testing (load model) 

## 1. prepare own model (.h5)
Make sure u put the models and know the model name, picture below

![](assets/11.PNG)

so for my model i just have my last models in logs/plant folder. My model name is mask_rcnn_custom_0050.h5

## 2. test own dataset
open load_model R50_E08.ipynb code in samples folder, then make sure the path is right and name of the model too.

![](assets/5.PNG)

Root_dir = (path) which u put whole files  (same as train)

![](assets/7.PNG)

dataset_train.load_data(Root_dir + your data train path )  (same as train)
dataset_val.load_data(Root_dir + your data val path )      (same as train)

![](assets/11.PNG)

model_path = Root_dir + (your own models)  

![](assets/10.PNG)
real_test_dir = Root_dir + (your test data path)    (same as train)

## 3. Execute all program
press shift+enter per line or Run all 



# Step to evaluate the model

## 1. evaluate model
so i make some evaluate you can just try and make the same path too like load model
evaluate include :
*IoU
*Dice
*mAP

## 2.Execute program
press shift+enter per line or Run all 


