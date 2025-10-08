# Creating Probability-Maximization-Based Clustering Targeting Model

Hi guys! In this tutorial, we will demonstrate on how to create cluster-based targeting model for your weapon systems.

For best results, please use:

* Fuzzy C-Means

  * Faster to converge than Expectation-Maximization.

  * Provides soft cluster memberships (players can influence multiple clusters).

  * Does not capture the spread of items.

* Expectation-Maximization

  * Captures the spread of items.

  * Produces probabilistic cluster assignments.

  * Slower to converge than Fuzzy C-Means.

## Initializing The Clustering Model

Before we can produce ourselves a targeting model, we first need to construct a model, which is shown below. Ensure that the distance function is not "CosineDistance".

```lua

local TargetingModel = DataPredict.Models.FuzzyCMeans.new({numberOfClusters = 3}) -- For this tutorial, we will assume that we have three missiles, so only three locations it can land.

```

## Collecting The Players' Locations

In order to find the center of the clusters, we first need all the players' location data and put them into a matrix.

```lua

local playerLocationDataMatrix = {

  {player1LocationX, player1LocationY, player1LocationZ},
  {player2LocationX, player2LocationY, player2LocationZ},
  {player3LocationX, player3LocationY, player3LocationZ},
  {player4LocationX, player4LocationY, player4LocationZ},
  {player5LocationX, player5LocationY, player5LocationZ},
  {player6LocationX, player6LocationY, player6LocationZ},
  {player7LocationX, player7LocationY, player7LocationZ},

}

```

## Getting The Center Of Clusters

Once you collected the players' location data, you must call model's train() function. This will generate the center of clusters to the model parameters.

```lua

TargetingModel:train(playerLocationDataMatrix)

```

Once train() is called, call the getModelParameters() function to get the center of cluster location data.

```lua

local centroidMatrix = TargetingModel:getModelParameters()

centroidMatrix = centroidMatrix[2] -- This is a must if you're using ExpectationMaximization because it stores the ModelParameters as a table of matrices.

```

## Interacting With The Center Of Clusters

Since we have three clusters, we can expect three rows for our matrix. As such we can process our game logic here.

```lua

for clusterIndex, unwrappedClusterVector in ipairs(ModelParameters) do

  local x = unwrappedClusterVector[1]
  
  local y = unwrappedClusterVector[2]
  
  local z = unwrappedClusterVector[3]

  landMissileAt(x, y, z)

end

```

## Resetting Our Targeting System

By default, when you reuse the machine learning models from DataPredict, it will interact with the existing model parameters. As such, we need to reset the model parameters by calling the setModelParameters() function and set it to "nil".

```lua

TargetingModel:setModelParameters(nil)

```

## Dual Approach

In this approach, we use Fuzzy C-Means to speed up the Expecation-Maximization model training. Because Fuzzy C-Means have lighter computations, you can perform more number of iterations per game frame.

```lua

local FastTargetingModel = DataPredict.Models.FuzzyCMeans.new({numberOfClusters = 3, maximumNumberOfIterations = 500})

local SlowTargetingModel = DataPredict.Models.ExpecationMaximization.new({numberOfClusters = 3, maximumNumberOfIterations = 30})

FastTargetingModel:train(playerLocationDataMatrix)

local meanMatrix = FastTargetingModel:getModelParameters()

local SlowModelParameters = {nil, meanMatrix}

FastTargetingModel:setModelParameters(SlowModelParameters)

FastTargetingModel:train(playerLocationDataMatrix)

local finalCentroidMatrix = TargetingModel:getModelParameters()[2] -- The final center locations.

```

That's all for today!
