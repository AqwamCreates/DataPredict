# Creating Cluster-Based Enemy Data Generation For Singleplayer Model

Hi guys! In this tutorial, we will demonstrate on how to create cluster-based enemy data generation model so that the enemies are based on previously defeated enemies' data.

For best results, please use:

* K-Means

  * This model sets up X number of clusters and finds the center of data.

* Fuzzy C-Means

  * This model sets up X number of clusters and finds the center of data.

* Agglomerative Hierarchical

  * This model sets up number of clusters that are equal to number of data and merge them together until it forms X number of clusters.

* MeanShift

  * This model sets up number of clusters that are equal to number of data and merge them together until it forms X number of clusters.
 
  * Trickier to set up.

## Initializing The Clustering Model

Before we can produce ourselves a difficulty generation model, we first need to construct a model, which is shown below. Ensure that the distance function is not "CosineDistance".

```lua

local EnemyDataGenerationModel = DataPredict.Models.KMeans.new({numberOfClusters = 1, distanceFunction = "Euclidean"}) -- For this tutorial, we will assume that we will generate one type of enemy.

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

Once you collected the players' combat data, you must call model's train() function. This will generate the center of clusters to the model parameters.

```lua

EnemyDataGenerationModel:train(playerCombatDataMatrix)

```

Once train() is called, call the getModelParameters() function to get the center of cluster location data.

```lua

local centerOfPlayerDataMatrix = EnemyDataGenerationModel:getModelParameters()

centerOfPlayerDataMatrix = centerOfPlayerDataMatrix[1] -- This is a must if you're using K-Means instead of Meanshift because K-Means stores the ModelParameters as a table of matrices.

```

## Generating The Enemy Data Using The Center Of Clusters

Since we have three clusters, we can expect three rows for our matrix. As such we can process our game logic here.

```lua

for clusterIndex, unwrappedCenterOfDataVector in ipairs(centerOfEnemyDataMatrix) do

  local playerBaseHealth = unwrappedCenterOfDataVector[1]
  
  local playerBaseDamage = unwrappedCenterOfDataVector[2]
  
  local playerBaseCashAmount = unwrappedCenterOfDataVector[3]

  local enemyHealth = playerBaseHealth * 0.5

  local enemyDamage = playerDamage * 0.1

  local enemyCashReward =  playerBaseCashAmount / 3

  spawnEnemy(enemyHealth, enemyDamage, enemyCashReward)

end

```

## Resetting Our Difficulty Generation System

By default, when you reuse the machine learning models from DataPredict, it will interact with the existing model parameters. As such, we need to reset the model parameters by calling the setModelParameters() function and set it to "nil".

```lua

DifficultyGenerationModel:setModelParameters(nil)

```

That's all for today!
