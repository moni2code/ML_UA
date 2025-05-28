import numpy as np 
import matplotlib.pyplot as plt
from numpy.random import rand
import seaborn as sns
import random
import torch 
import torchtext

def plot_grad_descent(lr=0.1, n_updates=10, x_scale=1):
  """
  Visualize gradient descent for simple linear regression.

  Args:
    lr (float, optional): Learning rate for gradient descent.
                          Default is 0.1.
    n_updates (int, optional): Number of gradient descent updates or iterations.
                               Default is 10.
    x_scale (float, optional): Scale parameter used for scaling random features.
                               Default is 1.

  Returns:
    None

  This function implements gradient descent for a simple linear regression 
  problem and visualizes the loss landscape and the path of weight updates.

  - `lr`: Controls the step size for weight updates during gradient descent.
  - `n_updates`: Specifies the number of iterations for gradient descent.
  - `x_scale`: Determines the scale for generating random feature values.

  Example usage:
  ```
  plot_grad_descent(lr=0.01, n_updates=20, x_scale=2)
  ```

  Note:
  This code uses Matplotlib for plotting and assumes that the necessary 
  libraries (e.g., NumPy and Matplotlib) are imported before calling the
   function.
  """


  """ 
  This function represents a simple linear regression model with a single
  feature x, weight weight, and an optional bias term b.
  """

  def uni_regression(x, weight, b=0):
    return weight*x + b

  """ 
  This function calculates the mean squared error (MSE) loss between the 
  predicted values y and the true values y_true.
  """

  def uni_loss(y, y_true):
    return np.mean((y - y_true)**2)

  """
  This function computes the gradient of the loss with respect to the weight 
  parameter w. It's used to update the weight during gradient descent.
  """

  def w_grad(y, y_true, x):
    return np.mean(2*(y - y_true)*x)

  """
  Here, you initialize the weight w, generate random values for the 
  feature x, and create an array y_true containing 100 ones. These values serve 
  as your initial conditions for gradient descent.
  """

  # w_opt = 1 
  w = -7
  x = x_scale*rand(100) 
  y_true = np.array(100*[1])

  loss, weights = [], []

  """
  This loop performs gradient descent for n_updates iterations. It computes
  the predicted values y, calculates the current loss, stores the weight and 
  loss values, and updates the weight using the gradient descent formula. 
  The learning rate lr controls the step size of the weight updates.
  """

  for i in range(0, n_updates):
    y = uni_regression(x, weight=w)
    current_loss = uni_loss(y, y_true)
    # print(current_loss)
    weights.append(w)
    loss.append(current_loss)
    dw = w_grad(y, y_true, x)
    # print("dw", dw)
    w = w - lr*dw
    # print(current_loss)
    if i == n_updates - 1:
        y = uni_regression(x, weight=w)
        current_loss = uni_loss(y, y_true)
        weights.append(w)
        loss.append(current_loss)

  loss_to_plot = []
  steps = np.linspace(-7, 8, num=100)

  for i in steps:
    loss_to_plot.append(uni_loss(uni_regression(x, weight=i), y_true))

  plt.plot(steps, loss_to_plot)

  plt.xlabel("W")
  plt.ylabel("MSE")

  """
  In this part, you create red arrows on the plot to visualize the path of 
  gradient descent. Each arrow represents the change in weight and loss from 
  one iteration to the next.
  """

  for i in range(1, len(weights)): 
    origin = np.array([[weights[i-1]],[loss[i-1]]])
    dw = weights[i] - weights[i-1]
    dloss = loss[i] - loss[i-1]
    V = np.array([[dw, dloss]])
    plt.quiver(*origin, V[:,0], V[:,1], color=['r'], scale=1, scale_units='xy', angles = 'xy')



def plot_mse(n_dots=10):
  """
  Visualize Mean Squared Error (MSE) error bars for a linear model.

  Args:
      n_dots (int, optional): Number of data points to visualize. Default is 10.

  Returns:
      None

  This function generates a plot to visualize the Mean Squared Error (MSE) error 
  bars for a linear model.

  - `n_dots`: Specifies the number of data points to visualize.

  Example usage:
  ```
  plot_mse(n_dots=20)
  ```

  """


  # Generate a linear function y = 2x + 2
  def lin_func(x):
      return 2*x + 2  

  sns.regplot(x=np.arange(0, 10), y=lin_func(np.arange(0, 10)))
  
  # Generate random x values
  x_random = [10*random.random() for i in np.arange(0, 10)]

  # Iterate to plot error bars and data points
  for i in np.arange(n_dots):
    x = 9*random.random()
    y = lin_func(x)
    delta = 10*(random.random() - 0.5)

    # Plot vertical lines representing the error bars
    plt.vlines(x, ymin=min(y, y + delta),  ymax=max(y, y + delta))
    # Plot data points with error (delta)
    plt.plot(x, y + delta,  marker='o', color='r')

  plt.xlabel("Your feature (Income)")
  plt.ylabel("Your target (Prices)")
  plt.title("MSE error bars for a linear model")
  plt.show()


def train_one_epoch(model, loss, optimizer, ds, l2_regularization):
  '''
  Train the given PyTorch model for one epoch.

  Args:
      model (torch.nn.Module): The PyTorch model to train.
      loss (torch.nn.Module): The loss function used for optimization.
      optimizer (torch.optim.Optimizer): The optimizer responsible for updating 
                                        model parameters.
      ds (torch.utils.data.Dataset): The PyTorch dataset used for training.
      l2_regularization (L2Regularization): l2 regularization object

  Returns:
      float: The average training loss for the epoch.

  This function trains a PyTorch model for one epoch using the provided dataset,
  loss function, and optimizer. It performs the following steps for each batch 
  in the dataset:

  1. Switches the model to the training mode.
  2. Initializes variables for accumulating the training loss and tracking the 
     dataset length.
  3. Iterates over batches in the dataset:
      - Transfers batch data to the GPU if available (assumes CUDA support).
      - Performs a forward pass through the model to obtain predictions.
      - Calculates the loss between predictions and ground truth labels.
      - Computes gradients with respect to the loss for model parameter updates.
      - Updates model parameters using the optimizer.
      - Records the batch's contribution to the training loss and updates the 
        dataset length.
      - Zeroes out the gradients in the optimizer.

  After processing all batches, the function returns the average training loss 
  for the epoch.

  Example usage:
  ```
  train_loss = train_one_epoch(model, loss_fn, optimizer, train_dataset)
  ```

  Note:
  This function assumes that the model, loss function, and optimizer are set up 
  appropriately before calling it.
  '''

  model.train() # we switch the model to the training mode
  train_loss = 0 # this variable accumulates loss
  ds_len = 0 # len of the dataset
  # loop over batches of the training set 
  for x, y in ds:
    # print("HERE", x.shape, y.shape)
    x, y = x.cuda(), y.cuda()
    output = model(x) # forward pass of the model 
    
    # we calculate loss and gradients for optimization
    l = loss(output, y)
    
    if l2_regularization:
      l+= l2_regularization.calculate(model)

    l.backward()

    # optimizer updates weights of the model 
    optimizer.step()

    # loss record 
    train_loss += l.item()*x.shape[0]
    ds_len += x.shape[0]
    optimizer.zero_grad()

  return train_loss/ds_len


def train(model, loss, val_metrics, optimizer, train_ds, dev_ds, num_epochs=10, 
          record_weights=None, early_stopper=None, l2_regularization=None):

  '''
  Train a PyTorch model over multiple epochs with optional early stopping and 
  record training metrics.

  Args:
      model (torch.nn.Module): The PyTorch model to train.
      loss (torch.nn.Module): The loss function used for optimization.
      val_metrics (dict): A dictionary of validation metrics to evaluate
                          during training. The keys are metric names, and the 
                          values are functions that compute the metrics.
      optimizer (torch.optim.Optimizer): The optimizer responsible for 
                                        updating model parameters.
      train_ds (torch.utils.data.Dataset): The PyTorch dataset used for 
                                          training.
      dev_ds (torch.utils.data.Dataset): The PyTorch dataset used for
                                        evaluation during training.
      num_epochs (int, optional): Number of training epochs. Default is 10.
      record_weights (function, optional): A function to record model weights 
                                          after each epoch. Default is None.
      early_stopper (EarlyStopper, optional): An EarlyStopper object for 
                                              implementing early stopping. 
                                              Default is None.

      l2_regularization (L2Regularization, optional): An L2Regularization object for 
                                              implementing early l2 regularization. 
                                              Default is None.



  Returns:
      dict: A dictionary containing training history, including training loss 
      and validation metrics.

  This function trains a PyTorch model over multiple epochs (with optional 
  early stopping based on validation metrics) using the provided dataset, loss 
  function, optimizer, and validation metrics. It records and returns the 
  training history, including training loss and validation metrics at each 
  epoch.


  - `model`: The PyTorch model to be trained.
  - `loss`: The loss function used for optimization.
  - `val_metrics`: A dictionary of validation metrics, where keys are metric 
                  names and values are functions.
  - `optimizer`: The optimizer responsible for updating model parameters.
  - `train_ds`: The training dataset.
  - `dev_ds`: The dataset used for evaluation during training.
  - `num_epochs`: Number of training epochs (default is 10).
  - `record_weights`: A function to record model weights after each epoch 
    (default is None).
  - `early_stopper`: If early stopping is used and triggered, the function loads
                     the best model weights and returns the training history. 
                     Otherwise, it returns the full training history.


  Example usage:
  ```
  history = train(model, loss_fn, validation_metrics, optimizer, 
                  train_dataset, dev_dataset, num_epochs=20)
  ```

  Note:
  - The `val_metrics` dictionary should contain metric functions that take model
    predictions and ground truth labels as input and return a scalar value.
  - The `record_weights` function is optional and can be used to record model 
    weights at each epoch.
  '''
  # here we record parameters of network after each epoch
  param_history = []

  # Dictionary to store training history
  history = {"train_loss": []}

  # Initialize storage for validation metrics
  for key in val_metrics:
    history['val_' + key] = []


  for epoch in range(num_epochs):

    if epoch == 0 and record_weights:
      param_history.append(record_weights(model))

    print('=========')
    # Train the model for one epoch and record the training loss
    current_train_loss = train_one_epoch(model=model, loss=loss, 
                                         optimizer=optimizer, ds=train_ds, 
                                         l2_regularization=l2_regularization)
    # Evaluate the model on the validation dataset and compute validation metrics
    val_metric_out = validate(model=model, val_metrics=val_metrics, ds=dev_ds)

    # Record validation metrics in the history dictionary
    for name, vm in val_metric_out.items():
      history[name].append(vm)

    history["train_loss"].append(current_train_loss)

    output2print = "epoch {}".format(epoch + 1) + \
                    " train loss: {:.4f} ".format(current_train_loss) + \
                    " ".join("{}: {:.4f}".format(k, v) for k, v in val_metric_out.items())

    if record_weights:
      param_history.append(record_weights(model))
                    
    print(output2print)


    if early_stopper is not None and early_stopper.early_stop(val_metric_out, model):
      print("EARLY STOPPING ")
      history["params"] = param_history
      model.load_state_dict(torch.load("best_model.pth"))
      return history
    
  return history


def test(model, ds):
  '''
  Make predictions using a PyTorch model on a dataset.

  Args:
      model (torch.nn.Module): The PyTorch model used for making predictions.
      ds (torch.utils.data.Dataset): The PyTorch dataset on which predictions 
                                      are made.

  Returns:
      torch.Tensor: A tensor containing the model's predictions for the dataset.

  This function evaluates the given PyTorch model on a dataset by making 
  predictions. It operates as follows:

  - Switches the model to evaluation mode using `model.eval()`.
  - Initializes an empty list called `final_output` to collect model predictions.
  - Iterates over batches in the test dataset:
      - Transfers batch data to the GPU if available (assumes CUDA support).
      - Sets `torch.no_grad()` to disable gradient calculations since this 
        is evaluation, not training.
      - Performs a forward pass through the model to obtain predictions.
      - Collects the predictions and appends them to the `final_output` list.

  After processing all batches, the function returns a tensor containing all the model's predictions for the dataset.

  Example usage:
  ```
  predictions = test(model, test_dataset)
  ```
  '''

  model.eval() # we switch the model to the evaluation mode
  final_output  = []
  # loop over batches of the test set
  for x, y in ds:
    x, y = x.cuda(), y.cuda()
    # we say that we do not want to calculate gradient for optimization
    with torch.no_grad():
      output = model(x) # forward pass of the model 
      # we collect all outputs of model
      final_output.append(output.detach().cpu())

  return torch.cat(final_output)



def validate(model, val_metrics, ds):
  '''
  Compute validation metrics for a PyTorch model on a validation dataset.

  Args:
      model (torch.nn.Module): The PyTorch model used for validation.
      val_metrics (dict): A dictionary of validation metrics to compute.
                          The keys are metric names, and the values are 
                          functions that compute the metrics.
      ds (torch.utils.data.Dataset): The PyTorch dataset used for validation.

  Returns:
      dict: A dictionary containing computed validation metrics.

  This function evaluates the given PyTorch model on a validation dataset and 
  computes specified validation metrics.It operates as follows:

  - Concatenates the ground truth labels from the validation dataset to form a 
    single tensor `y_test`.
  - Calls the `test` function to obtain model predictions for the validation 
    dataset, stored in `model_pred`.
  - Initializes an empty dictionary `metric_out` to store computed validation 
    metrics.
  - Iterates over the provided validation metrics:
      - Computes each metric by calling the corresponding metric function with 
        `model_pred` and `y_test`.
      - Stores the computed metric in the `metric_out` dictionary with a prefix
         'val_' added to the metric name.

  After processing all validation metrics, the function returns a dictionary 
  containing the computed validation metrics.

  Example usage:
  ```
  validation_metrics = {
      'accuracy': accuracy_function,
      'loss': loss_function
  }
  metrics = validate(model, validation_metrics, validation_dataset)
  ```

  Note:
  - The returned dictionary contains the computed validation metrics, and the metric names are prefixed with 'val_' for clarity.
  '''

  y_test = torch.cat([y for x, y in ds])

  model_pred = test(model, ds)
  metric_out = {}

  for name, metric in val_metrics.items():
    metric_out['val_' + name] = metric(model_pred, y_test)
  
  return metric_out



class EarlyStopper:
  def __init__(self, metric_name, patience=1, min_delta=0, minimize=True):
    '''
    Initialize an EarlyStopper object for monitoring a validation metric.

    Args:
        metric_name (str): The name of the validation metric to monitor (e.g., 'accuracy', 'loss').
        patience (int, optional): The number of epochs to wait for improvement before early stopping. Default is 1.
        min_delta (float, optional): The minimum change required in the metric to be considered as improvement. Default is 0.
        minimize (bool, optional): If True, aim to minimize the metric; if False, aim to maximize it. Default is True.

    Attributes:
        patience (int): The number of epochs to wait for improvement before early stopping.
        min_delta (float): The minimum change required in the metric to be considered as improvement.
        counter (int): Counter to keep track of the number of epochs without improvement.
        min_val_metric (float): The minimum observed value of the validation metric.
        max_val_metric (float): The maximum observed value of the validation metric.
        name (str): The name of the validation metric being monitored.
        minimize (bool): Whether to minimize the metric (True) or maximize it (False).

    This class is used to implement early stopping based on the specified validation metric.
    '''
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.min_val_metric = None
    self.max_val_metric = None
    self.name = metric_name
    self.minimize = minimize

  def min_criteria(self, val_metric, model):

    '''
    Check if training should be stopped based on minimizing the validation metric.

    Args:
        val_metric (float): The value of the validation metric for the current epoch.
        model (torch.nn.Module): The PyTorch model being trained.

    Returns:
        bool: True if training should be stopped; False otherwise.
    '''

    print(val_metric, self.min_val_metric)

    if self.min_val_metric is None or val_metric < self.min_val_metric:
        self.min_val_metric = val_metric
        self.counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    elif val_metric > (self.min_val_metric + self.min_delta):
        self.counter += 1
        if self.counter >= self.patience:
            return True
    return False

  def max_criteria(self, val_metric, model):
    '''
    Check if training should be stopped based on maximizing the validation metric.

    Args:
        val_metric (float): The value of the validation metric for the current epoch.
        model (torch.nn.Module): The PyTorch model being trained.

    Returns:
        bool: True if training should be stopped; False otherwise.
    '''

    if self.max_val_metric is None or val_metric > self.max_val_metric:
        self.max_val_metric = val_metric
        self.counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    elif val_metric < (self.max_val_metric - self.min_delta):
        self.counter += 1
        if self.counter >= self.patience:
            return True
    return False


  def early_stop(self, val_metrics, model):
    '''
    Determine if early stopping criteria are met based on the validation metric.

    Args:
        val_metrics (dict): A dictionary containing validation metrics (e.g., {'val_accuracy': 0.85}).
        model (torch.nn.Module): The PyTorch model being trained.

    Returns:
        bool: True if training should be stopped; False otherwise.

    This method checks whether early stopping criteria are met based on the validation metric.
    '''

    val_metric = val_metrics['val_' + self.name]
    if self.minimize:
      return self.min_criteria(val_metric, model)
    else:
      return self.max_criteria(val_metric, model)



class L2Regularization:
  def __init__(self, l2_lambda=0.01):
      self.l2_lambda = l2_lambda


  def calculate(self, model):
    l2_reg = torch.tensor(0.,  device='cuda:0')
    for param in model.parameters():
        l2_reg += torch.norm(param) 

    return l2_reg*self.l2_lambda



def build_embed_matrix(reverse_word_index, dim=50, num_words=5000):
  """
  Build an embedding matrix using pre-trained GloVe word vectors and/or random embeddings.

  Args:
      reverse_word_index (list or dict): A list or dictionary mapping integer indices to words in the vocabulary.
      dim (int, optional): The dimensionality of word embeddings (default is 50).
      num_words (int, optional): The maximum number of words to consider for embedding (default is 5000).

  Returns:
      torch.Tensor: A 2D tensor representing the embedding matrix, where each row corresponds to a word embedding.

  Note:
      This function uses pre-trained GloVe word vectors to create an embedding matrix for a given vocabulary.
      If a word is found in the GloVe vocabulary, its pre-trained embedding is used; otherwise, a random
      embedding is generated.

  Example:
      # Example usage:
      reverse_word_index = {1: 'apple', 2: 'banana', 3: 'cherry'}
      embedding_matrix = build_embed_matrix(reverse_word_index, dim=100, num_words=4)
  """

  glove = torchtext.vocab.GloVe('6B', dim=dim)

  embedding_matrix = np.zeros((num_words, dim)) 
  counter = 0

  emb_mean = glove.vectors.mean()
  emb_std =  glove.vectors.std()

  for i in range(1, num_words):
    # print()
    
    word = reverse_word_index[i]
  
    if word in glove.stoi or word.lower() in glove.stoi:
       embedding_matrix[i] = glove.get_vecs_by_tokens([word], 
                                                      lower_case_backup=True)
    else:
      embedding_matrix[i] = np.random.normal(emb_mean, emb_std, size=(1, dim))
      counter +=1

  return torch.tensor(embedding_matrix)


