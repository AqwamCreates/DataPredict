# What is Distributed Training?

Distributed training is a way to train main model parameters from child model parameters that are derived from the main one. This is useful if you wish to train each model with their own data but would like to merge them together later. This could lead to increased training speed or better generalization.

There are two types of distributed training classes contained in this library:

* DistributedGradients

* DistributedModelParameters

Below, I will explain what each of these do and show you how to use these classes below. But first, we need to create multiple models and train them first.

```lua

-- Let's initialize 3 LinearRegression models here.

local LinearRegression = DataPredict.Models.LinearRegression

local LinearRegression1 = LinearRegression.new()

local LinearRegression2 = LinearRegression.new()

local LinearRegression3 = LinearRegression.new()

-- Then, we will train them here. Let's assume we know the datasets of featureMatrix and labelVector for each model.

LinearRegression1:train(featureMatrix1, labelVector1)

LinearRegression2:train(featureMatrix2, labelVector2)

LinearRegression3:train(featureMatrix3, labelVector3)

```

## DistributedGradients

For DistributedGradients, the calculated gradients from child model parameters are sent to the main model parameters. Only applicable for:

  * LinearRegression
    
  * LogisticRegression
    
  * NeuralNetwork

I will show you how to use DistributedGradients in the sample code shown below.

```lua

-- First, let's initialize our DistributedGradients object here.

local DistributedGradients = DataPredict.Models.DistributedGradients.new()

-- Then we need a model parameters from a model and send it to the DistributedGradients object.

local ModelParameters1 = LinearRegression1:getModelParameters()

DistributedGradients:setModelParameters(ModelParameters1)

-- For this to work, we need to change some parameters for the LinearRegression objects.
-- I will only set parameters for one model, so let's assume I also did this to other models.

LinearRegression1:setAreGradientsSaved(true) -- We need to save the gradients for every iterations, so we set this true.

LinearRegression1:setParameters(1) -- We also need to make the number of iterations to 1.

-- Once set, we can start training our models individually and update the model parameters in DistributedGradients object.

LinearRegression1:train(featureVector1, labelScalar1)

local Gradients1 = LinearRegression:getGradients()

DistributedGradients:addGradients(Gradients1)

-- addGradients() will update the model parameters in DistributedGradients object.
-- Once updated, you can call DistributedGradients' getModelParameters() to update the LinearRegression's model parameters.

local UpdatedModelParameters = DistributedGradients:getModelParameters()

LinearRegression1:setModelParameters(UpdatedModelParameters)

```

## DistributedModelParameters

For DistributedModelParameters, the child model parameters are combined to create new main model parameters.

Just like the DistributedGradients, I will show you how to use DistributedModelParameters.

```lua

-- First, let's initialize our DistributedModelParameters object here.

local DistributedModelParameters = DataPredict.Models.DistributedModelParameters.new()

-- Second, we need to initialize our ModelParametersMerger object and put it into the DistributedModelParameters object.

local ModelParametersMerger = DataPredict.Models.ModelParametersMerger.new()

DistributedModelParameters:setModelParametersMerger(ModelParametersMerger)

-- Then we need a model parameters from a model and send it to the DistributedModelParameters object.

local ModelParameters1 = LinearRegression1:getModelParameters()

DistributedModelParameters:setMainModelParameters(ModelParameters1)

-- For this to work, we need to change some parameters for the LinearRegression objects.
-- I will only set parameters for one model, so let's assume I also did this to other models.

LinearRegression1:setAreGradientsSaved(false) -- We don't need to save the gradients because we're directly using the model parameters.

LinearRegression1:setParameters(500) -- We need to set the number of iterations to certain values so the cost values converges.

-- We then need to add the models to DistributedModelParameters.

DistributedModelParameters:addModel(LinearRegression1)

-- Once set, we can start training our models individually and update the model parameters in DistributedModelParameters object.

DistributedModelParameters:train(featureVector1, labelScalar1, 1)

-- The train() or reinforce() functions from DistributedModelParameters will update the main model parameters in DistributedModelParameters
-- object when the number of train() or reinforce() function calls reaches certain limits.
-- Once updated, you can call DistributedGradients' getMainModelParameters() to update the LinearRegression's model parameters.

local UpdatedModelParameters = DistributedModelParameters:getMainModelParameters()

LinearRegression1:setModelParameters(UpdatedModelParameters)

```

## Conclusion

The code samples might seem complex for setting up distributed training classes at first, but with practice, you'll find it much easier to set up.

That's all for today!
