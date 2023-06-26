# simplegrad

## A basic, small autograd engine that backpropagates over a Directed Acyclic Graph (DAG). Based on the neural network tutorials provided by Andrej Kaparthy. 

This engine can employ a deep neural network to solve binary classification problems. It currently does this by using ReLu for hidden layers and a sigmoid (base logistic) function for the output layer (assuming classification is 0 or 1). Should functionality require classification of (-1,1), keyword `activation` argument for the MLP class can be changed to `tanh`. There is scope is add functionality to alter hidden layer non-linearity function.

## Simple implementation

### Setting up the data

I will demonstrate a usecase whereby a user can employ a neural net to solve a binary classification exercise. This will make use of certain regularisation techniques such as L1 reguluralisation and dropoff regularisation. Dropoff is enabled by the set_training_mode() function within the MLP and Neuron classes. Let's begin!

```python
from simple_nn import *
from simple_mlp import *
import numpy as np
import pandas as pd

data = pd.read_excel('somethingsomething.xlsx')
```

Assuming our data is in a numpy array, we can assign features and labels in the following way. This also assumes the first column are the labels. The code may change depending on the location of the label (usually the last column) and if it is a pandas dataframe.

We can then create a training and validation set for dropoff purposes

```python
labels = data[:, 0]
features = data[:,1:]

from sklearn.utils import class_weight
import random
random.seed(1337)

# Calculate the class weights
class_weights = class_weight.compute_sample_weight('balanced', y = labels)

# Set up sizes
data_size = len(features)
train_frac = 0.8  # Fraction of data to use for training
train_size = int(data_size * train_frac)  # Number of examples to use for training

# Shuffling data indices
indices = list(range(data_size))
random.shuffle(indices)

# Split indices
train_indices = indices[:train_size]
valid_indices = indices[train_size:]

# Split features and labels into training and validation sets
train_features = [[Value(feat) for feat in features[i]] for i in train_indices]
train_labels = [Value(labels[i]) for i in train_indices]
valid_features = [[Value(feat) for feat in features[i]] for i in valid_indices]
valid_labels = [Value(labels[i]) for i in valid_indices]
```

### Optimising the loss function

We first intialise an MLP object, and define our loss function. You can modify this loss function with class weights if necessary. An tiny value named epsilon (or whatever you like) can be added to ensure we are not trying to take the log of a zero value -> math domain error

```python
# Initialize your model
model = MLP(54, [10, 1])  # MLP with 54 input neurons, one hidden layer of 10 neurons and one output neuron

def compute_binary_cross_entropy(labels, output):
    epsilon = 1e-10
    return sum([Value(-y.data) * (y_hat+epsilon).log() 
                - Value(1-y.data) * ((1-y_hat+epsilon).log()) 
                for y, y_hat in zip(labels, output)])
    

# Define learning rate and number of epochs
initial_lr = 0.01
decay_rate = 0.5
decay_steps = 10
epochs = 200
```

We then can proceed to train the model
```python
# Training loop
for epoch in range(epochs):
    model.set_training_mode(True)  # Enable dropout
    
    # Training phase
    train_output = [model(feat_list) for feat_list in train_features]
    train_loss = compute_binary_cross_entropy(train_labels, train_output)
    alpha = 1e-4
    reg_loss = alpha * sum((abs(p.data) for p in Function.flatten(model.parameters())))
    total_train_loss = train_loss + reg_loss
    
    # Backward pass
    Function.zero_grad(model)
    total_train_loss.backward()
    
    # Update parameters
    lr = initial_lr * (decay_rate ** (epoch // decay_steps))
    for p in Function.flatten(model.parameters()):
        p.data -= initial_lr * p.grad
    
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss.data}")
    
    # Validation phase
    model.set_training_mode(False)  # Disable dropout
    valid_output = [model(feat_list) for feat_list in valid_features]
    valid_loss = compute_binary_cross_entropy(valid_labels, valid_output)
    
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {valid_loss.data}")
```

### Model evaluation
Once our model has been trained, we can set_training_mode(False) and make predictions. 
```python
model.set_training_mode(False)
# get predictions on training data
train_output = [model(feat_list) for feat_list in train_features]
train_predictions = [1 if prob.data > 0.5 else 0 for prob in train_output]

# get predictions on validation data
valid_output = [model(feat_list) for feat_list in valid_features]
valid_predictions = [1 if prob.data > 0.5 else 0 for prob in valid_output]
```

We can then implement the following code to assess the performance. Repeat for valid set and test set (also repeat above on test data to obtain test predictions)
```python
train_accuracy = sum([1 if label.data == pred else 0 for label, pred 
                      in zip(train_labels, train_predictions)]) / len(train_labels)

print(f"Training Accuracy: {train_accuracy}")

true_positives = sum([1 if label.data == pred == 1 else 0 for label, pred in zip(train_labels, train_predictions)])
false_positives = sum([1 if label.data == 0 and pred == 1 else 0 for label, pred in zip(train_labels, train_predictions)])
false_negatives = sum([1 if label.data == 1 and pred == 0 else 0 for label, pred in zip(train_labels, train_predictions)])

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

f1_score = 2 * ((precision * recall) / (precision + recall))
print(f"Training Precision: {precision}")
print(f"Training Recall: {recall}")
print(f"Training F1-score: {f1_score}")
```

### Saving and loading the model
If you are happy with the results (make sure you have assessed training, valid, and test!) then you can go ahead and save the model. In fact even if the model performs poorly you can save the model to use later for model comparison analysis. 

```python
import dill

# Save the model
with open('model_v7.pkl', 'wb') as f:
    dill.dump(model, f)

# Load the model
with open('model_v7.pkl', 'rb') as f:
    loaded_model = dill.load(f)
```
You can then use this loaded_model to predict on new unseen data like below
```python
test_out = [loaded_model(feat) for feat in test_features]
test_predictions = [1 if prob.data > 0.5 else 0 for prob in test_out]
```

## Concluding thoughts
This has been an interesting project adapting it to a specific usecase. Will look to improve generality of the model and application to other usecases in the future.











