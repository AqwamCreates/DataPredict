# Creating Probability-Based NPC Anti-Clump Model

Hi guys! In this tutorial, we will demonstrate on how to create cluster-based NPC anti-clump model.

For best results, please use Expectation-Maximization model.

## Initializing The Clustering Model

```lua

 -- For this tutorial, we will let the model decide how many clusters it will produce based on player / item spread.

-- Note, we're setting math.huge here, but that doesn't mean we will begin producing an infinite amount of clusters! It will start at 1 and increases it until the model finds a suitable number of clusters.

local PlacementModel = DataPredict.Models.ExpectationMaximization.new({numberOfClusters = math.huge})

```

## Collecting The Players' Locations

In order to find the center of the clusters, we first need all the players' location data and put them into a matrix.

```lua

local npcLocationDataMatrix = {

  {npc1LocationX, npc1LocationY, npc1LocationZ},
  {npc2LocationX, npc2LocationY, npc2LocationZ},
  {npc3LocationX, npc3LocationY, npc3LocationZ},
  {npc4LocationX, npc4LocationY, npc4LocationZ},
  {npc5LocationX, npc5LocationY, npc5LocationZ},
  {npc6LocationX, npc6LocationY, npc6LocationZ},
  {npc7LocationX, npc7LocationY, npc7LocationZ},

}

```

## Getting The Center Of Clusters

Once you collected the NPCs' location data, you must call model's train() function. This will generate the center of clusters to the model parameters.

```lua

PlacementModel:train(npcLocationDataMatrix)

```

Once train() is called, call the getModelParameters() function to get the center of cluster location data.

```lua

local centroidMatrix = PlacementModel:getModelParameters()

centroidMatrix = centroidMatrix[1]

```

## Interacting With The Center Of Clusters

Since we have dynamic number of clusters, we can expect multiple rows for our matrix. As such we can process our game logic here.

```lua

local directionStrength = 0.5

local function goToPlayer(NPC, Player)

  local meanMatrix = ModelParameters[1]
  
  local varianceMatrix = ModelParameters[2]
  
  local numberOfClusters = #meanMatrix
  
  local randomClusterIndex = math.random(1, numberOfClusters)

  local randomUnwrappedMeanVector = meanMatrix[randomClusterIndex]

  local randomUnwrappedVarianceVector = varianceMatrix[randomClusterIndex]

  local randomX = randomUnwrappedMeanVector[1] + ((math.random() * 2 - 1) * randomUnwrappedVarianceVector[1])

  local randomY = randomUnwrappedMeanVector[2] + ((math.random() * 2 - 1) * randomUnwrappedVarianceVector[2])

  local randomZ = randomUnwrappedMeanVector[3] + ((math.random() * 2 - 1) * randomUnwrappedVarianceVector[3])

  local playerX, playerY, playerZ = getPlayerPosition(Player)

  local npcX, npcY, npcZ = getNPCPosition(NPC)

  local directionX = playerX - randomX

  local directionY = playerY - randomY

  local directionZ = playerZ - randomZ

  local x = npcX + (directionStrength * directionX)

  local y = npcY + (directionStrength * directionY)

  local z = npcZ + (directionStrength * directionZ)

  moveNPCto(NPC, x, y, z)

end

```

## Resetting Our Placement System

By default, when you reuse the machine learning models from DataPredict, it will interact with the existing model parameters. As such, we need to reset the model parameters by calling the setModelParameters() function and set it to "nil".

```lua

PlacementModel:setModelParameters(nil)

```

That's all for today!
