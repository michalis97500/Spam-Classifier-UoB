
## Introduction

This project was an assignment during my Masters course at The University of Bath on February 2022.
Course: MSc Computer Science Module: Artificial Intelligence

The aim of the assignment is to construct spam classifier model using machine learning in python. The model was trained with data provided by the University.

## Model

### Neural network
A neural network is used to create the model. The neural network contains the following methods
> 1) init() -> Class constructor
> 2) calcAccuracy(features, labels) -> Calculates accuracy of a set of data by predicting results of features and comparing to labels, returns the result
> 3) predict (array) -> Makes a prediction and returns it
> 4) Link() -> Links the current layer to the previous layer
> 5) fwdPropagation(training_features, training_labels) -> Function that aids the training of the network. Returns the accuracy using calcAccuracy
> 6) backPropagation(labels) ->  Using the labels, this method calculates delta Y of the cost funcion and adds weights and biases to the layers
> 7) layerPrepare() -> Initialises the layers if any of them do not currently have weights and/or biases assigned
> 8) train(epochs, batchSize -> Using the backPropagation function, this method is used to train the neural network
> 9) getTrainBatch(batch, batchsize) -> Method to get the data used for training. Returns tuple -> (features, labels)
> 10) setTreinData(features, labels) -> Helper function to set the objects (self) features and labels
> 11) save(filename) -> saves weighs and biases as a file (JSON format)
> 12) load(filename) -> loads layer weights and biases from file (JSON format)

### Layers
This class is used for the creation of layers in the neural network. It contains several methods of calculating cost. A good background for what each function does can be found here 
https://www.simplilearn.com/tutorials/machine-learning-tutorial/cost-function-in-machine-learning#:~:text=For%20neural%20networks%2C%20each%20layer,is%20called%20the%20global%20minima.

## Result 

This is the feedback from the University
>Code feedback
>Classifier accuracy on hidden data is: 92.6%
Your estimate was 92.0% which is basically spot on!
Part one marks (out of 50)
38.3/45 marks for the classifier accuracy on hidden data
5.0/5 marks for the estimate
Total part one: 43/50

>Video feedback
>Overall grade: 40/50 
>Extra comments: This is a strong overview of your NN approach to this problem. 
>Grading categories/areas for improvement:
> Solution complexity: ★★★★☆ 
> Demonstrates understanding of theory and application: ☐ Flawed/Insufficient ☐ Basic ☑ Strong ☐ Thorough 
> Shows results: ☑ 
> Results put in context: ☐ Explains decisions and parameters: ☑ 
> Presentation/polish: ★★★★☆ 
> Note that grade is not directly calculated from these categories, they are just for feedback. More information and general feedback will be posted on the forum

The project achieved a final grade of 83/100
