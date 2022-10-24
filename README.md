# Neural_Network_Charity_Analysis
Module 19

## Overview of the Analysis
Alphabet Soup, which is a philanthropic organization with a mission to help other organizations which protect the envionment, imporve people's wellbeing, and unify the world. This organization has raised over 10 billion dollars over the course of 20 years and distrubutes charitable funds to organizations developing technology and also to reforestation groups internationally. The purpose of this project is to analyze the impact of donations and to vet potential recipients for charity. In order to do this, a mathematical/data-driven solution is required to predict which organizations are worth donating to and which ones are too high risk. This project utilizes a deep learning neural network which evaluates a variety of input data relevant to previously funded organizations and produces a decision-making model. This deep learning neural network can then be used to predict if a future donation is high risk or not.

This analysis utilizes the Python TensorFlow library to build and optimize the deep neural network.

## Results
The following section answers questions concerning the data preprocessing, deep learning model building, model evaluation, and model optimization.

### Data Preprocessing
In this portion of the project, the `charity_data.csv` file was read and used to make a dataframe. Non-beneficial columns (variables) were removed from the dataframe. Columns containing categorical variables with a large number of unique values were modified with binning (combining groups together). These categorical variables underwent binary encoding and reincorporation into the original dataframe. This modified dataset was then split into feature and target arrays, then split into training and testing groups, scaled, and finally input into a deep neural network model.

#### What variable(s) are considered the target(s) for your model?

#### What variable(s) are considered to be the features for your model?

#### What variable(s) are neither targets nor features, and should be removed from the input data?

### Compiling, Training, and Evaluating the Model

#### How many neurons, layers, and activation functions did you select for your neural network model, and why?

#### Were you able to achive the target model performance?

#### What steps did you take to try and increase model performance?

## Summary
