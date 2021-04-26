# NeuroEvolution
## Purpose ##
NeuroEvolution is a Python Implementation of an elementary TWEANN with the capabilities of evolving the weights of a fixed neural network, links for each weight, a recurrent layer, and activation functions for each layer. 

- A link is an associated binary value, 0 or 1, which allows the network to only allow certain nodes to influence the model

## Documentation
Note that no error checking has been implemented so if it fails; your inputs for the arguments are incorrect.
- There are currently 7 activation functions available: 'relu', 'tanh', 'logistic', 'unit', 'softmax', 'gaussian', 'purlin'
    - relu: max(0,x)
    - tanh: (e^x - e^-x)/(e ^x + e^-x)
    - logistic (sigmoid): 1/(1+e^-x)
    - unit: 0 if x <= 0 ; 1 x > 0
    - softmax: e^x_i / sum(e^x_i)
    - gaussian: e^ (-x^2)
    - purlin (identity): x

### Constructor 
#### Arguments
- layer_nodes : (list) whose length equals number of layers and each value represents the number of nodes
```python
# example
layer_nodes = [5,10,2]  # three layer 5-10-2 node network
```
- num_input : (int) represents the number of input variables
```python
# example
num_input = 6  # 6 input variables
```
- num_output : (int) represents the number of output variables
```python
# example
num_output = 1  # 1 output variables
```
- default_activation_function : (str) represents the default activation function for all hidden layers; can either be a single string or list of strings (must equal length of number of hidden layers)
```python
# example
default_activation_function = 'relu' # same for all layers
default_activation_function = ['relu', 'logistic'] # specify for different layers
```
- default_output_activation : (str) represents the default activation function for output layer
```python
# example
default_output_function = 'purlin' # only for output
```
### Evolve
- For evolve_activation_layer and evolve_recurrent_layer, the activation functions chosen for these lists represents the different possible activation functions available for their respecive layer

#### Arguments
- input : (np array) the input, X, vector to train upon (number of columns must equal previously inputted number of inputs)
```python
# example
input = np.random.uniform(-1,1, 100*6).reshape(100,6)  # 100x6 
```
- expected_output : (np array) the expected output, Y, vector to train upon 
```python
# example
expected_output = np.random.uniform(-1,1, 100).reshape(100,1)  # 100x1
```
- error_function : (str) the expected error function to evalute the error between predicted and expected_output. Can be either 'mse', 'mae', or 'r_2'
    - mse : Mean Squared Error
    - mae : Mean Absolute Error
    - r_2 : R^2, not correlation coefficient, 1-MSS/TSS
```python
# example
error_function = 'mse'  # mean sequared error
```
- max_generations : (int) maximum number of generations to evalute
```python
# example
max_generations = 100
```
- max_gen_size : (int) number of initial individuals to create
```python
# example
max_gen_size = 1000
```
- use_prev : (bool) evolve from previous evolution (NOT YET IMPLEMENTED)
```python
# example
use_prev = False  # evolve from scratch
use_prev = True  # evolve from previous generation
```
- info : (bool) print out diagnostic output
```python
# example
info = True  # print out diagnostic
info = False # do not print out dignostic
```
- tol : (double) the tolerance to exit the algorithm if the current best minus the previous best is under this value after 25 generations
```python
# example
tol = 1e-10
```
- evolve_links : (bool) evolve the links of the network or not
```python
# example
evolve_links = True  # evolve links
evolve_links = False # do not evolve links
```
- evolve_activation_layer : (list) if not None, evolve the activation functions for each layer, must be a list of strings like default_activation_function in the constructor
```python
# example
evolve_activation_layer = None # do not evolve activation 
evolve_activation_layer = ['relu', 'logistic']  # length must equal that of hidden layers
```
- evolve_recurrent_layer : (list) if not None, evolve the recurrent layer, must be a list of strings like default_activation_function in the constructor
```python
# example
evolve_recurrent_layer = None # do not evolve activation 
evolve_recurrent_layer = ['relu', 'logistic']  # length must equal that of hidden layers
```
### Predict
- input : (numpy array) the input, X, vector to predict upon (number of columns must equal previously inputted number of inputs)

- avg : (bool) if True, predict using the average of the best three unique models; if False, predict using the best model

### get_score
Returns a list of scores from either the best model or average of the best three unique models. In order, returns [R^2, MAE, MSE]; if links are used, appends [num_links_used, total_possible_links] where both of those values are a two element list, first entry denoting links for weights and biases of hidden layer, second for links of recurrent layer
- avg : (bool) if True, return average of best three unique models; if False, return scores for best model
