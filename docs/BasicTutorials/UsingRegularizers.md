# What Is A Regularizer?

Regularizers are used to ensure that the model do not overfit. In other words, we want to ensure that the model to generalize instead of memorizing solutions.

# Getting Started

In order to show how regularization works, we need to introduce a single dataset

```lua

-- A simple linear trend: Y = X + 10.

-- But we add one "noisy" data point at the end to trick the model.

-- Input (X) -> Expected Output (Y)'

-- 1 -> 11
-- 2 -> 12
-- 3 -> 13
-- 4 -> 14
-- 5 -> 15
-- 6 -> 100  <-- THE TRAP! (This is the noise/outlier)

local featureMatrix = { -- The column of 1 is for bias so that it can learn to add +10.

  	{1, 1},
	{1, 2},
	{1, 3},
	{1, 4},
	{1, 5},
	{1, 6} 

}

local labelVector = {

	{11},
	{12},
	{13},
	{14},
	{15},
	{100} -- This outlier forces the non-regularized model to "memorize" this error.

}

```

In order for us to use the regularization, we need to create an regularizer object.

# Memorized Vs Generalized Pattern Prediction

In here, we will show you the comparisons between using and not using a regularizer.

```lua

local LinearRegression = DataPredict.Models.LinearRegression

local MemorizedLinearRegression = LinearRegression.new()

local ElasticNetRegularizer = DataPredict.Regularizer.ElasticNet.new() # This is our regularizer.

local GeneralizedLinearRegression = LinearRegression.new({Regularizer = ElasticNetRegularizer}) # The regularizer is placed into this model.

```

In here, this is where our training starts.

```lua

MemorizedLinearRegression:train(featureMatrix, labelVector)

GeneralizedLinearRegression:train(featureMatrix, labelVector)

```

This is where how we determine if the model "memorized" the data's pattern.

## In Outlier Case

```lua

local featureVectorFromExistingFeatureMatrix = {{1, 6}}

local memorizedLabelValue = MemorizedLinearRegression:predict(featureVectorFromExistingFeatureMatrix)[1][1]

local generalizedLabelValue = GeneralizedLinearRegression:predict(featureVectorFromExistingFeatureMatrix)[1][1]

print(memorizedLabelValue, generalizedLabelValue) -- You'll notice that the memorized label value is closer to the original value.

```

## In Normal Case

```lua

local featureVectorFromExistingFeatureMatrix = {{1, 1}}

local memorizedLabelValue = MemorizedLinearRegression:predict(featureVectorFromExistingFeatureMatrix)[1][1]

local generalizedLabelValue = GeneralizedLinearRegression:predict(featureVectorFromExistingFeatureMatrix)[1][1]

print(memorizedLabelValue, generalizedLabelValue) -- You'll notice that the generalized label value is closer to the original value.

```

That's all for today! Do note that not all models uses regularizer, so please check the API reference if this option is available or not.

That's all for now!
