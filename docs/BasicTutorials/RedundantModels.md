# What Are Redudant Models

This may be surprising to you, but most of the DataPredict's models that handles regression are actually redundant. Though, those models were added for users' convenience as well as "fluffing up my model count".

To show what I mean, I'll generate a dataset that best captures this.

```lua

-- A simple time-to-leave trend: Y = X^2.

-- Input (X) -> Expected Output (Y).

--[[

  Notice that these are positive-only label values.
  Hence, you cannot use linear regression as it will predict negative label values.

--]]

-- 1 -> 1
-- 2 -> 4
-- 3 -> 9
-- 4 -> 16
-- 5 -> 25
-- 6 -> 36

local featureMatrix = {

  {1},
  {2},
  {3},
  {4},
  {5},
  {6},

}

local labelVector = {

  {1},
  {4},
  {9},
  {16},
  {25},
  {36},

}

```

# Equivalent Models

In here, you can give your linear regression to have gamma regression properties provided that you perform these steps:

```lua

local StrangeLinearRegression = DataPredict.Models.LinearRegression.new()

-- 1. Convert your label vector so that it becomes log(Y).

local modifiedLabelVector = TensorL2D:applyFunction(math.log, labelVector)

-- 2. Train your model using this modified label vector.

StrangeLinearRegression:train(featureMatrix, modifiedLabelVector)

-- 3. Predict using a test feature matrix.

local modifiedPredictedLabelVector = StrangeLinearRegression:predict({{3}})

-- 4. Convert it back using the inverse function of logarithm, which is the exponent.

local predictedLabelVector = TensorL2D:applyFunction(math.exp, modifiedPredictedLabelVector)

print(predictedLabelVector[1][1]) -- We get 9!

```

Let's compare this with our Gamma Regression:

```lua

local GammaRegression = DataPredict.Models.GammaRegression.new()

-- 1. Train your model using the original label vector.

GammaRegression:train(featureMatrix, labelVector)

-- 2. Predict using a test feature matrix.

local predictedLabelVector = GammaRegression:predict({{3}})

print(predictedLabelVector[1][1]) -- We get 9!

```

Gasp! They're the same! So it is redundant!

But wait! It gets better!

Linear regression requires multiple number of iterations, just like gamma regression. Let's grab a closed-form model: ridge regression.

Suddenly, it becomes an instant solution!

```lua

local StrangeRidgeRegression = DataPredict.Models.RidgeRegression.new()

-- 1. Convert your label vector so that it becomes log(Y).

local modifiedLabelVector = TensorL2D:applyFunction(math.log, labelVector)

-- 2. Train your model using this modified label vector.

StrangeRidgeRegression:train(featureMatrix, modifiedLabelVector)

-- 3. Predict using a test feature matrix.

local modifiedPredictedLabelVector = StrangeRidgeRegression:predict({{3}})

-- 4. Convert it back using the inverse function of logarithm, which is the exponent.

local predictedLabelVector = TensorL2D:applyFunction(math.exp, modifiedPredictedLabelVector)

print(predictedLabelVector[1][1]) -- We get 9!

```

# Investigating The Model Parameters

If you use getModelParameters(), you will see that these models contains the same weights (generally).

```lua

local GammaRegressionModelParameters = GammaRegression:getModelParameters()

local StrangeLinearRegressionModelParameters = StrangeLinearRegression:getModelParameters()

local StrangeRidgeRegressionModelParameters = StrangeRidgeRegression:getModelParameters()

TensorL2D:printTensor(GammaRegressionModelParameters, StrangeLinearRegressionModelParameters, StrangeRidgeRegressionModelParameters) -- Gasp! They're the same as well!

```

# Conclusion

As you can see, all the fancy regression models can be reduced two different parts: linear regression and data transformation.

This means that you can also replace binary regression with linear regression as well, but to keep the tutorial simple, I chose gamma regression due to only requiring logarithm and exponent values handling.

That's all I have to show for today. As you can see, you all were sold a lie that you need specialized models for handling complex cases.

I just proved you otherwise. Keep this in mind.

Keep it simple. Keep it instant.

Nothing can beat a closed-form linear regression with some data transformation.

Suddenly, you realize the lines between a machine learning engineer, a data scientist and a statistician begin to blur. This is basically the combination of these three roles' knowledge.
