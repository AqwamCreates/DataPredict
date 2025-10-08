# Creating Probability-Based Clustering Placement Model

Hi guys! In this tutorial, we will demonstrate on how to create cluster-based placement model.

For best results, please use Expectation-Maximization model.

## Initializing The Clustering Model

```lua

 -- For this tutorial, we will let the model decide how many clusters it will produce based on player / item spread.

-- Note, we're setting math.huge here, but that doesn't mean we will begin producing an infinity amount of clusters! It will start at 1 until it finds a suitable number of clusters.

local PlacementModel = DataPredict.Models.ExpectationMaximization.new({numberOfClusters = math.huge})

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

local function placePlayerAtARandomLocation(Player)

  local meanMatrix = ModelParameters[1]
  
  local varianceMatrix = ModelParameters[2]
  
  local numberOfClusters = #meanMatrix
  
  local randomClusterIndex = math.random(1, numberOfClusters)

  local randomUnwrappedMeanVector = meanMatrix[randomClusterIndex]

  local randomUnwrappedVarianceVector = varianceMatrix[randomClusterIndex]

  local x = randomUnwrappedMeanVector[1] + (math.random() * randomUnwrappedVarianceVector[1])

  local y = randomUnwrappedMeanVector[2] + (math.random() * randomUnwrappedVarianceVector[2])

  local z = randomUnwrappedMeanVector[3] + (math.random() * randomUnwrappedVarianceVector[3])

  placePlayer(Player, x, y, z)

end

```

## Resetting Our Placement System

By default, when you reuse the machine learning models from DataPredict, it will interact with the existing model parameters. As such, we need to reset the model parameters by calling the setModelParameters() function and set it to "nil".

```lua

PlacementModel:setModelParameters(nil)

```

That's all for today!
