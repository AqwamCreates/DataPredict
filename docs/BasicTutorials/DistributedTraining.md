# What is Distributed Training?

Distributed training is a way to train main model parameters from child model parameters that are derived from the main one. This is useful if you wish to train each model with their own data but would like to merge them together later. This could lead to increased training speed or better generalization.

There are two types of distributed training classes contained in this library:

* DistributedGradientsCoordinator

* DistributedModelParametersCoordinator

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

I will show you how to use DistributedGradientsCoordinator in the sample code shown below.

```lua

-- First, let's initialize our DistributedGradientsCoordinator object here.

local DistributedGradientsCoordinator = DataPredict.DistributedTrainingStrategies.DistributedGradientsCoordinator.new()

-- Then we need a model parameters from a model and send it to the DistributedGradients object.

local ModelParameters1 = LinearRegression1:getModelParameters()

DistributedGradientsCoordinator:setModelParameters(ModelParameters1)

-- For this to work, we need to change some parameters for the LinearRegression objects.
-- I will only set parameters for one model, so let's assume I also did this to other models.

LinearRegression1:setAreGradientsSaved(true) -- We need to save the gradients for every iterations, so we set this true.

LinearRegression1:setParameters({maximumNumberOfIterations = 1}) -- We also need to make the number of iterations to 1.

-- Once set, we can start training our models individually and update the model parameters in DistributedGradientsCoordinator object.

LinearRegression1:train(featureMatrix1, labelVector1)

local Gradients1 = LinearRegression:getGradients()

DistributedGradientsCoordinator:addGradients(Gradients1)

-- addGradients() will update the model parameters in DistributedGradientsCoordinator object.
-- Once updated, you can call DistributedGradientsCoordinator's getModelParameters() to update the LinearRegression's model parameters.

local UpdatedModelParameters = DistributedGradientsCoordinator:getModelParameters()

LinearRegression1:setModelParameters(UpdatedModelParameters)

```

## DistributedModelParametersCoordinator

For DistributedModelParametersCoordinator, the child model parameters are combined to create new main model parameters.

Just like the DistributedGradientsCoordinator, I will show you how to use DistributedModelParametersCoordinator.

```lua

-- First, let's initialize our DistributedModelParametersCoordinator object here.

local DistributedModelParametersCoordinator = DataPredict.DistributedTrainingStrategies.DistributedModelParametersCoordinator.new()

-- Second, we need to initialize our ModelParametersMerger object and put it into the DistributedModelParametersCoordinator object.

local ModelParametersMerger = DataPredict.Models.ModelParametersMerger.new()

DistributedModelParametersCoordinator:setModelParametersMerger(ModelParametersMerger)

-- For this to work, we need to change some parameters for the LinearRegression objects.
-- I will only set parameters for one model, so let's assume I also did this to other models.

LinearRegression1:setAreGradientsSaved(false) -- We don't need to save the gradients because we're directly using the model parameters.

LinearRegression1:setParameters({maximumNumberOfIterations = 500}) -- We will set the maximum number of the iterations to 500 for this tutorial.

-- Then we train our model.

LinearRegression1:train(featureMatrix, labelVector)

-- We then need to add the trained model parameters to DistributedModelParameters.

local TrainedModelParameters1 = DistributedModelParametersCoordinator:getModelParameters()

DistributedModelParametersCoordinator:addModelParameters(TrainedModelParameters1)

-- The addModelParameters() from DistributedModelParametersCoordinator will update the main model parameters.
-- Once updated, you can call DistributedModelParametersCoordinator' getMainModelParameters() to update the LinearRegression's model parameters.

local UpdatedModelParameters = DistributedModelParametersCoordinator:getMainModelParameters()

LinearRegression1:setModelParameters(UpdatedModelParameters)

```

## Conclusion

The code samples might seem complex for setting up distributed training classes at first, but with practice, you'll find it much easier to set up.

That's all for today!
