It is important for us to check the accuracy of our model. Hence, in this tutorial we will show you on how to check your model's test and validation costs.

# Getting Started

First, we need to create a linear regression model and a model checker objects for testing and validation. We will set the second parameter as "regression" for the ModelChecker because the linear regression model falls under the "regrression".

```
local Model = MDLL.Model.LinearRegression.new()

local ModelChecker = MDLL.Others.ModelChecker.new(Model, "regression")
```

# Testing

Right now, we will test our LogisticRegression model for test cost.  We will also provide testFeatureMatrix and testLabelVector as well.

```
local testCost = ModelChecker:test(testFeatureMatrix, testLabelVector)
```

The above function will generate the accuracy of the model by comparing the predicted output made by the model and the actual value.

Then use print() to see the test cost.

```
print(testCost)
```

# Validation

Validation is similar to testing, but instead requires two pairs of featureMatrix and labelVector. The output we get are the training and validation cost arrays. 

```
local trainCostArray, validationCostArray = ModelChecker:testClassificationModel(trainFeatureMatrix, trainLabelVector, validationFeatureMatrix, validationLabelVector)
```

The above function will generate the accuracy of the model by comparing the predicted output made by the model and the actual value.

Then use print() to see the train and validation cost arrays.

```
print(trainCostArray)

print(validationCostArray)
```

That's all for today!
