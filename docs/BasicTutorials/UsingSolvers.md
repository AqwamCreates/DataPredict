# What Are Solvers?

Solvers are the "engines" behind our models. They determine **how** the model calculates its answers. Some solvers are fast but approximate (like taking a shortcut), while others are slow but mathematically perfect (like measuring every step).

Choosing the right solver depends on your dataset size and whether you need an instant answer or a highly precise one.

# Getting Started

To see the difference, we will use a very small, clean dataset. This allows even the slowest solvers to finish quickly so we can compare them.

```lua

-- A simple linear trend: Y = 2*X + 5.

-- Input (X) -> Expected Output (Y).

-- 1 -> 7
-- 2 -> 9
-- 3 -> 11
-- 4 -> 13
-- 5 -> 15

-- The column of 1 is for bias (the +5).

local featureMatrix = { 

    {1, 1},
    {1, 2},
    {1, 3},
    {1, 4},
    {1, 5},
}

local labelVector = {

    {7},
    {9},
    {11},
    {13},
    {15},

}

```

# Training

In this comparison, we will test two different types of solvers:

* Gauss-Newton Solver: Directly finds the next best solution based on current model parameters. Generally requires the number of datapoints to be greater than the number of features in the feature matrix. This is generally the default setting for most models.

* Gradient Solver: It starts with a random guess and slowly improves it step-by-step. It takes longer but can handle massive datasets that would crash the Instant solver.

Note that all solvers uses the "residual form" and not the "complete form". In other words, improvements are made based on the difference between the true label value and predicted label value.

```lua

local LinearRegression = DataPredict.Models.LinearRegression

local GaussNewtonSolver = DataPredict.Solvers.GaussNewton.new() 

local GradientSolver = DataPredict.Solvers.Gradient.new()

local GaussNewtonModel = LinearRegression.new({Solver = GaussNewtonSolver})

local GradientModel = LinearRegression.new({Solver = GradientSolver})

```

Now, let's train both models. Note that the models prints out the cost for each number of iterations.

```lua

GaussNewtonModel:train(featureMatrix, labelVector)

GradientModel:train(featureMatrix, labelVector)

```

Notice will notice that Gauss-Newton solver converges faster when compared to the gradient solver.

# Prediction

Despite the different solvers, they both should give nearly identical results for this dataset.

```lua

local testFeatureVector = {{1, 10}}

local gaussNewtonLabelValue = GaussNewtonModel:predict(testFeatureVector)[1][1]

local gradientLabelValue = GradientModel:predict(testFeatureVector)[1][1]

print(gaussNewtonLabelValue, gradientLabelValue) -- Both should be very close to 25!

```

That's all for now! Remember, the solver that you choose must be based on your goal. Though, if you are attempting to train incrementally, it is recommended to use gradient solver.

However, I'll warn you that the solvers may predict different label values if you are using a dirty / noisy feature matrix despite the solvers are using same dataset.

Also, DataPredict actually have cache system that speeds up calculations. So, if you are worried that certain solvers may be computationally expensive, the cache is always stored for linear models.

Lastly, check the API reference to see which models are able to use the solvers.
