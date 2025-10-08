# Creating Probability-Based Clustering Placement Model

Hi guys! In this tutorial, we will demonstrate on how to create cluster-based placement model.

For best results, please use Expectation-Maximization model.

## Initializing The Clustering Model

```lua

local PlacementModel = DataPredict.Models.ExpectationMaximization.new({numberOfClusters = 3}) -- For this tutorial, we will assume that we have three missiles, so only three locations it can land.

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

PlacementModel:train(playerLocationDataMatrix)

```

Once train() is called, call the getModelParameters() function to get the center of cluster location data.

```lua

local centroidMatrix = PlacementModel:getModelParameters()

centroidMatrix = centroidMatrix[1]

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

PlacementModel:setModelParameters(nil)

```

That's all for today!
