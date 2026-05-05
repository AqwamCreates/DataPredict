# What Is Data Transformation

Data transformation is basically means that we add, delete, merge or modify the values inside our dataset so that our models can train and predict better.

It technically allows you to use models that are not designed for the original dataset, provided that you maintain consistency on how you handle the modified dataset.

# Redudant Models

This may be surprising to you, but most of the models that handles regression are actually redundant. Though, those models were added for users' convenience as well as "fluffing up my model count".

To show what I mean, I'll generate a dataset that best captures this.

```lua

-- A simple time-to-leave trend: Y = (XW)^2.

-- Input (X) -> Expected Output (Y).

-- Notice that these are positive-only label values. Hence, you cannot use linear regression as it will predict negative label values.

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

## Equivalent Models

In here, you can give your linear regression to have gamma regression properties provided that you perform these steps:

```lua

local StrangeLinearRegression = DataPredict.Models.LinearRegression.new()

-- 1. Convert your label vector so that it becomes log(Y).

local modifiedLabelVector = TensorL2D:applyFunction(math.log, labelVector)

-- 2. Train your model using this modified label vector.

StrangeLinearRegression:train(featureMatrix, modifiedLabelVector)

-- 3. Predict using a test feature matrix.

local modifiedPredictedLabelVector = StrangeLinearRegression:train({{3}})

-- 4. Convert it back using the inverse function of log.

local predictedLabelVector = TensorL2D:applyFunction(math.exp, modifiedPredictedLabelVector)

print(predictedLabelVector[1][1]) -- We get 9!

```

Let's compare this with our Gamma Regression:

```lua

local GammaRegression = DataPredict.Models.GammaRegression.new()

-- 1. Train your model using the original label vector.

GammaRegression:train(featureMatrix, labelVector)

-- 2. Predict using a test feature matrix.

local predictedLabelVector = GammaRegression:train({{3}})

print(predictedLabelVector[1][1]) -- We get 9!

```

Gasp! They're the same! So it is redundant!

But wait! It gets better!

Linear regression requires multiple number of iterations, just like gamma regression. Let's grab a closed-form model: ridge regression.

Suddenly, it becomes instant solution!

```

local StrangeRidgeRegression = DataPredict.Models.RidgeRegression.new()

-- 1. Convert your label vector so that it becomes log(Y).

local modifiedLabelVector = TensorL2D:applyFunction(math.log, labelVector)

-- 2. Train your model using this modified label vector.

StrangeRidgeRegression:train(featureMatrix, modifiedLabelVector)

-- 3. Predict using a test feature matrix.

local modifiedPredictedLabelVector = StrangeRidgeRegression:train({{3}})

-- 4. Convert it back using the inverse function of log.

local predictedLabelVector = TensorL2D:applyFunction(math.exp, modifiedPredictedLabelVector)

print(predictedLabelVector[1][1]) -- We get 9!

```
