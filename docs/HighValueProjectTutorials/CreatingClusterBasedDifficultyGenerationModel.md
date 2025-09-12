# Creating Cluster-Based Difficulty Generation Model

Hi guys! In this tutorial, we will demonstrate on how to create cluster-based difficulty generation model so that the enemies are not too easy or too hard for everyone.

For best results, please use:

* K-Means or Sequential K-Means if you want to manually set the number of clusters.

* MeanShift if you want the model to set the number of clusters. (Trickier to set up)

## Initializing The Clustering Model

Before we can produce ourselves a difficulty generation model, we first need to construct a model, which is shown below. Ensure that the distance function is not "CosineDistance".

```lua

local DifficultyGenerationModel = DataPredict.Models.KMeans.new({numberOfClusters = 1, distanceFunction = "Euclidean"}) -- For this tutorial, we will assume that we will generate one type of enemy.

```

## Collecting The Players' Combat Data

In order to find the center of the clusters, we first need all the players' combat data and put them into a matrix.

```lua

local playerCombatDataMatrix = {

  {player1MaximumHealth, player1MaximumDamage, player1CashAmount},
  {player2MaximumHealth, player2MaximumDamage, player2CashAmount},
  {player3MaximumHealth, player3MaximumDamage, player3CashAmount},

}

```

## Getting The Center Of Clusters

Once you collected the players' combat data, you must call K-Means' train() function. This will generate the center of clusters to the model parameters.

```lua

DifficultyGenerationModel:train(playerCombatDataMatrix)

```

Once train() is called, call the getModelParameters() function to get the center of cluster location data.

```lua

local ModelParameters = DifficultyGenerationModel:getModelParameters()

```

## Generating Difficulty Using The Center Of Clusters

Since we have three clusters, we can expect three rows for our matrix. As such we can process our game logic here.

```lua

for clusterIndex, unwrappedClusterVector in ipairs(ModelParameters) do

  local playerBaseHealth = unwrappedClusterVector[1]
  
  local playerBaseDamage = unwrappedClusterVector[2]
  
  local playerBaseCashAmount = unwrappedClusterVector[3]

  local enemyHealth = playerBaseHealth * 0.5

  local enemyDamage = playerDamage * 0.1

  local enemyCashReward =  playerBaseCashAmount / 3

  summonEnemy(enemyHealth, enemyDamage, enemyCashReward)

end

```

## Resetting Our Difficulty Generation System

By default, when you reuse the machine learning models from DataPredict, it will interact with the existing model parameters. As such, we need to reset the model parameters by calling the setModelParameters() function and set it to "nil".

```lua

DifficultyGenerationModel:setModelParameters(nil)

```

That's all for today!
