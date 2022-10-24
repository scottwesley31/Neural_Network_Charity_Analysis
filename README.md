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
Here is how the neural network was defined in the **first optimization** attempt:
![opt1_nn](https://user-images.githubusercontent.com/107309793/197434043-b919b9e2-e1be-490a-8dd9-ab755e7c758d.png)
In this case, I did not change the number of neurons (80 and 30), layers (2), or activation functions ("relu" for hidden and "sigmoid" for output) from the original model design.

I kept the neurons and layers the same mostly because I wanted to check and see if I could improve the accuracy of the model first by modifying the input data.

The first hidden layer has 80 neurons in it because it is following the good rule of thumb to have two to three times the amount of neurons in the hidden layer as number of inputs. After the data is preprocessed and undergoes binary encoding, the number of inputs = 41.

The number of neurons in the second layer is more arbitrary. It's moreso important that a second layer exists to give the neural network more of a chance to identify nonlinear characteristics.

The Rectified Linear Unit (ReLU) function felt appropriate for the input data because of it's simplifiying output and flexibility (returning a value from 0 to infinity).

The sigmoid function was selected for the output layer since it is classifying the `IS_SUCCESSFUL` result as either 0 or 1 (yes or no).

Here is how the neural network was defined in the **second optimization** attempt:
![opt2_nn](https://user-images.githubusercontent.com/107309793/197436438-98a09c3c-95d7-4d6e-b3f0-fbf9ae34041b.png)

In the second attempt, the only change made was the addition of a 3rd hidden layer with 30 neurons. The selection of 30 neurons within the 3rd layer was mostly arbitrary again. The third layer was added to see if it made a difference in model accuracy.

The `relu` activation function was also maintained in this example to provide the same simplification and flexibility executed in the previous layers.

Here is how the neural network was defined in the **third optimization** attempt:
![opt3_nn](https://user-images.githubusercontent.com/107309793/197437080-99763974-5706-4ff7-b8f7-38898d4688cd.png)

In the third optimization attempt, it was determined that an additional hidden layer did not have a significant impact on the accuracy score of the model, so it was removed. The number of neurons in the first hidden layer was increased to 168 (which was calculated by multiplying the number of input variables - in this case 42 - by 3; 42 was also added to this number in an attempt to account for bias terms but this was an improper rationale in retrospect).

A different activation function was used in the hidden layers (`tanh`) mostly to see if this would change the accuracy score but also to see if output ranges between -1 and 1 could better represent the patterns in this dataset. It was observed after scaling the training and testing data that some negative values did exist, leading to the decision to use `tanh` and to potentially better account for these values.

#### Were you able to achive the target model performance?
Here are the associated model loss and model accuracy scores obtained for each attempt.

**Original Attempt**
![original_accuracy](https://user-images.githubusercontent.com/107309793/197439485-00d69619-2e96-49a1-95b0-029f4fb453bd.png)

**Optimization Attempt 1**
![opt1_accuracy](https://user-images.githubusercontent.com/107309793/197439529-d51f26f5-e1d2-4195-828d-4b935380fa29.png)

**Optimization Attempt 2**
![opt2_accuracy](https://user-images.githubusercontent.com/107309793/197439548-10ba3ab0-3e1b-4062-83d6-bfdc41d6ef77.png)

**Optimization Attempt 3**
![opt3_accuracy](https://user-images.githubusercontent.com/107309793/197439574-27372605-e7dd-4d8f-badb-fcec3f322fdd.png)

To summarize:
- Original: Loss of **0.64**, Accuracy of **0.61**
- Optimization Attempt 1: Loss of **0.84**, Accuracy of **0.64**
- Optimization Attempt 2: Loss of **0.80**, Accuracy of **0.53**
- Optimization Attempt 3: Loss of **0.67**, Accuracy of **0.69**

I was unable to reach an accuracy score of **0.75 (75%)** but did improve the model accuracy by **8%** (from 0.61 to 0.69). Loss increases were also minimized (from 0.64 to 0.67; only 3% loss).

#### What steps did you take to try and increase model performance?
Here is a list of the changes I made in each optimization attempt

**Optimization Attempt 1**:
- Dropped `SPECIAL_CONSIDERATIONS` column from input data

**Optimization Attempt 2**:
- Dropped `SPECIAL_CONSIDERATIONS` and `STATUS` column from input data
- Decreased `APPLICATION_TYPE` bins from 9 to 7 (more unique values under `Other` category)
- Dropped `AFFILIATION_Other` and `USE_CASE_Other` from encoded dataframe prior to scaling (found outliers in scaled data from previous round in these columns by looking at a box plot of the `X_train_scaled` data)
- Added a 3rd hidden layer with 30 neurons

**Optimization Attempt 3**:
- Dropped `ASK_AMT` from input data after determining this might be a noisy variable with a lot of outliers (looked at boxplot of all features).
- Kept `SPECIAL_CONSIDERATIONS` and `STATUS` columns within input data.
- Kept binning for `APPLICATION_TYPE` at 7 categories.
- Increased number of neurons in the first hidden layer 168 (at least triple the number of inputs)
- Changed the activation functions for the first and second hidden layers to `tanh` to potentially account for more negative input/output values.

## Summary
From the original model design to the first optimization attempt: **loss increased by 20% (0.64 to 0.84)** and **accuracy increased by 3% (0.61 to 0.64)** after removing a potentially non-beneficial column from the features.

From the first optimization attempt to the second: **loss decreased by 4% (0.84 to 0.80)** and **accuracy decreased by 11% (0.64 to 0.53)** after removing 2 input columns, modifying binning, removing columns from encoded data, and adding a 3rd hidden layer. This drop in accuracy may have been due to overfitting.

From the second optimization attempt to the third: **loss decreased by 13% (0.80 down to 0.67)** and **accuracy increased by 16% (0.53 to 0.69)** after keeping columns dropped in the 2nd attempt, dropping a potentially noisy variable, maintaining binning as before, increasing the number of neurons substantially in the first hidden layer, and changing the hidden layer activation functions from `relu` to `tanh`.

**Recommendation:** The need to generate a model which can predict whether or not it is worth it for Alphabet Soup to donate money to organizations is a classification problem. It would definitely be worth it to compare the results of a **random forest classifier** on this dataset. It could be beneficial to split the data up into weak learners during the training phase, rather than relying on the evaluation of input data within neurons accross multiple layers. Using the random forest classifier would also save a lot of time during the optimization phase of this project since there are less parameters to modify during the training step.
