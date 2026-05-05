# What Are Optimizers?

Optimizers are deep learning techniques that adjusts our machine/deep learning model learning rates. They make our models train faster and hence require less number of iterations.

# Getting Started

In order to show how optimizers works, we need to introduce a single dataset.

```lua

-- A simple linear trend: Y = X + 10.

-- Input (X) -> Expected Output (Y).

-- 1 -> 11
-- 2 -> 12
-- 3 -> 13
-- 4 -> 14
-- 5 -> 15
-- 6 -> 16

local featureMatrix = { -- The column of 1 is for bias so that it can learn to add +10.

	{1, 1},
	{1, 2},
	{1, 3},
	{1, 4},
	{1, 5},
	{1, 6},

}

local labelVector = {

	{11},
	{12},
	{13},
	{14},
	{15},
	{16},

}

```

# Unoptimized Vs Optimized Training

In here, we will show you the comparisons between using and not using an optimizer.

Note that it is recommended to use optimizers with the gradient solvers. This is because optimizers are redundant for other solvers

```lua

local GradientSolver = DataPredict.Solvers.Gradient

local GradientSolver1 = GradientSolver.new()

local GradientSolver2 = GradientSolver.new()

local LinearRegression = DataPredict.Models.LinearRegression

local UnoptimizedLinearRegression = LinearRegression.new({Solver = GradientSolver1})

local AdaptiveGradientOptimizer = DataPredict.Optimizer.AdaptiveGradient.new() # This is our optimizer.

local OptimizedLinearRegression = LinearRegression.new({Solver = GradientSolver2, Optimizer = AdaptiveGradientOptimizer}) # The optimizer is placed into this model.

```

In here, this is where our training starts. Note that the models prints out the cost for each number of iterations.

```lua

UnoptimizedLinearRegression:train(featureMatrix, labelVector)

OptimizedLinearRegression:train(featureMatrix, labelVector)

```

Notice that the unoptimized model converges far more slowly compared to the optimized one. The optimizer basically speeds up the model's learning process by reducing the number of iterations needed to converge.

# GradientClippers And ValueShedulers Can Be Optimizers Too!

Because the way DataPredict designed the gradient clippers and value schedulers to be similar to optimizers, we can adjust the model's cost function derivatives without the need for the optimizer themselves.

That's all for now! Do note that not all models uses optimizers, so please check the API reference if this option is available or not.
