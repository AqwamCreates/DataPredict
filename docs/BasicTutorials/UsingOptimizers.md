# What Are Optimizers?

Optimizers are deep learning techinques that adjusts our machine/deep learning model learning rates. They make our models train faster and hence require less number of iterations.

# Getting Started

# Getting Started

In order to show how optimizers works, we need to introduce a single dataset.

```lua

-- A simple linear trend: Y = X + 10.

-- But we add one "noisy" data point at the end to trick the model.

-- Input (X) -> Expected Output (Y).

-- 1 -> 11
-- 2 -> 12
-- 3 -> 13
-- 4 -> 14
-- 5 -> 15

local featureMatrix = { -- The column of 1 is for bias so that it can learn to add +10.

	{1, 1},
	{1, 2},
	{1, 3},
	{1, 4},
	{1, 5},

}

local labelVector = {

	{11},
	{12},
	{13},
	{14},
	{15},

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



Then, we can now train with our optimizer included. Do note that not all models uses optimizers, so please check the API reference if this option is available or not.

# GradientClippers And ValueShedulers Can Be Optimizers Too!

Because the way we designed the gradient clippers and value schedulers to be similar to optimizers, we can adjust the model's cost function derivatives without the need for the optimizer themselves.

That's all for now!
