It is important for us to check the performance of our model. Hence, in this tutorial we will show you on how to check your model's test and validation costs.

# Getting Started

First, we need to create a linear regression model and a model checker objects for testing and validation. We will set the second parameter as "regression" for the ModelChecker because the linear regression model falls under the "regrression".

```lua
local LinearRegression = DataPredict.Model.LinearRegression.new()

local ModelChecker = DataPredict.Others.ModelChecker.new({Model = LinearRegression, modelType = "Regression"})
```

# Testing

Right now, we will test our LogisticRegression model for test cost.  We will also provide testFeatureMatrix and testLabelVector as well.

```lua
local testCost = ModelChecker:test(testFeatureMatrix, testLabelVector)
```

The above function will generate the accuracy of the model by comparing the predicted output made by the model and the actual value.

Then use print() to see the test cost.

```lua
print(testCost)
```

# Validation

Validation is similar to testing, but instead requires two pairs of featureMatrix and labelVector. The output we get are the training and validation cost arrays. 

```lua
local trainCostArray, validationCostArray = ModelChecker:validate(trainFeatureMatrix, trainLabelVector, validationFeatureMatrix, validationLabelVector)
```

The above function will generate the accuracy of the model by comparing the predicted output made by the model and the actual value.

Then use print() to see the train and validation cost arrays.

```lua
print(trainCostArray)

print(validationCostArray)
```

That's all for today!
