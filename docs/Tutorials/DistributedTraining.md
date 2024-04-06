# What is Distributed Training?

Distributed training is a way to train main model parameters from child model parameters that are derived from the main one. This is useful if you wish to train each model with their own data but would like to merge them together later. This could lead to increased training speed or better generalization.

There are two types of distributed training classes contained in this library:

* DistributedGradients - The calculated gradients from child model parameters are sent to the main model parameters.

* DistributedModelParameters - The child model parameters are combined to create new main model parameters.

Below, I will show you how to use these classes below. But first, we need to create multiple models and train them first.

```lua

-- Let's initialize 3 LinearRegression models here.

local LinearRegression = DataPredict.Models.LinearRegression

local LinearRegression1 = LinearRegression.new()

local LinearRegression2 = LinearRegression.new()

local LinearRegression3 = LinearRegression.new()

-- Then, we will train them here. Let's assume we know the dataset of featureMatrix and labelVector.

LinearRegression1:train(featureMatrix, labelVector)

LinearRegression2:train(featureMatrix, labelVector)

LinearRegression3:train(featureMatrix, labelVector)

```

## DistributedGradients
