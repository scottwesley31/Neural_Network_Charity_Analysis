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
Here is a screenshot of all of the columns (variables) found in the `charity_data.csv` file:
![variables](https://user-images.githubusercontent.com/107309793/197432111-42fbb79b-fd16-4179-b4d1-d9f9096cfe6c.png)

Since the purpose of this analysis is to determine which organizations are worth donating to from past data, the only variable that makes sense as a target is the `IS_SUCCESSFUL` column. This column indicates whether or not a particular donation went to good use or not. This is what we want the model to predict from future inputs.

#### What variable(s) are considered to be the features for your model?
Most of the other variables would be considered features. There are a few variables that may not have any actual affect on the dependent `IS_SUCCESSFUL` - like `EIN` and `NAME`.

#### What variable(s) are neither targets nor features, and should be removed from the input data?
As stated above, there are a few variables that may not have an affect on whether or not the donation was successful. These include `EIN` and `NAME` which are simply identification columns that are arbitrary. Some other columns that were considered later on in analysis as potential unimpactful features were `STATUS` and `SPECIAL_CONSIDERATIONS`.

### Compiling, Training, and Evaluating the Model
In this section of the project, the deep neural net was defined, trained, and evaluated. The number of input features, hidden nodes, hidden layers, and activation functions were defined prior to compiling the model and training it. Saving checkpoints (for every 5 epochs) were added as the model was trained. The accuracy and loss were then calculated after feeding testing data into the neural net.

#### How many neurons, layers, and activation functions did you select for your neural network model, and why?
Here is how the neural network was defined in the first optimization attempt:
![opt1_nn](https://user-images.githubusercontent.com/107309793/197434043-b919b9e2-e1be-490a-8dd9-ab755e7c758d.png)
In this case, I did not change the number of neurons (80 and 30), layers (2), or activation functions ("relu" for hidden and "sigmoid" for output) from the original model design.

I kept the neurons and layers the same mostly because I wanted to check and see if I could improve the accuracy of the model first by modifying the input data.

The first hidden layer has 80 neurons in it because it is following the good rule of thumb to have two to three times the amount of neurons in the hidden layer as number of inputs. After the data is preprocessed and undergoes binary encoding, the number of inputs = 41.

The number of neurons in the second layer is more arbitrary. It's moreso important that a second layer exists to give the neural network more of a chance to identify nonlinear characteristics.

The Rectified Linear Unit (ReLU) function felt appropriate for the input data because of it's simplifiying output and flexibility (returning a value from 0 to infinity).

The sigmoid function was selected for the output layer since it is classifying the `IS_SUCCESSFUL` result as either 0 or 1 (yes or no).

#### Were you able to achive the target model performance?

#### What steps did you take to try and increase model performance?

## Summary
