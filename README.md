# PicturesPrediction

Using the dataset of CIFAR10, which is a set about pictures of vehicles and animals and has 10 classes. The main goal is to predict for each image of our testing set (so it's from CIFAR10 always) to which label it is associated. This is done in pytorch and keras to see if there are differences. 
There is a grid displaying a few images from this dataset : 
![ImagesOnGrid](https://user-images.githubusercontent.com/74551760/217904265-cbaa25e3-7c4d-4913-ba2a-4c4c7c4c773f.PNG)

## First CNN
We try to develop a first convolutionnal neural network, to have a first idea.
After that we try to optimize it adding BatchNorm, Dropout, more filters in convolutions etc and get around 69% acccuracy. 

## Data Augmentation 
The principle here is to take each images from our training set only, to apply a few transformation on them and to make the network learn it to be better. 
For example we take a picture of a truck, we apply a rotation but we don't change the label it's always a truck. There is an example afer. 

Normal car picture : 

![ExampleCar](https://user-images.githubusercontent.com/74551760/217904748-2a50f67a-1216-46d2-8bed-9f93611146f8.png)

Same picture but reversed : 

![ExampleCarReverse](https://user-images.githubusercontent.com/74551760/217904757-0e8e9404-6424-4370-8f8e-aa693860dc68.png)

The result is clearly better but not suffisant i think. We are around 88 % accuracy. 

## Transfer learning 
Here we just have to take a pre-trained network (in our case it's ResNet50), take the output of the layer before the real output (because it's only a label and we want a picture) and put it in output of our neural network. We are at a better level of accuracy here and get around 93 %. 
