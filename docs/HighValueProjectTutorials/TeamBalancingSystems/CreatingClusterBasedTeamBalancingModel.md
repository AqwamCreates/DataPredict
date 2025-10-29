# Creating Cluster-Based Team Balancing Model

Hi guys! In this tutorial, we will demonstrate on how to create cluster-based team balancing model so that no team can outcompete the others.

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

* Expectation-Maximization

  * This model sets up X number of clusters and finds the center of data.
 
  * Slow convergence, but extremely likely to find what the player wants.

## Initializing The Clustering Model

Before we can produce ourselves a team balancing model, we first need to construct a model, which is shown below. Ensure that the distance function is not "CosineDistance".

```lua

-- For this tutorial, we will assume that we will generate two clusters that have similar player data.

local TeamBalancingModel = DataPredict.Models.KMeans.new({numberOfClusters = 2, distanceFunction = "Euclidean"}) 

```

## Collecting The Players' Combat Data

In order to find the center of the clusters, we first need all the players' combat data and put them into a matrix.

```lua

local playerIndexMapping = {player1, player2, player3}

local playerCombatDataMatrix = {

  {player1KillDeathRatio, player1ScorePerKill},
  {player2KillDeathRatio, player2ScorePerKill},
  {player3KillDeathRatio, player3ScorePerKill},

}

```

## Getting The Center Of Clusters

Once you collected the players' combat data, you must call model's train() function. This will generate the center of clusters to the model parameters.

```lua

TeamBalancingModel:train(playerCombatDataMatrix)

```

Once train() is called, call the predict() function to assign players to individual clusters.

```lua

local assignedClusterNumberVector = TeamBalancingModel:predict(playerCombatDataMatrix)

```

Then we add custom logic where we assign teams to each players. 

In here, we're making sure we're alternating clusters because each clusters have similar player data

```

local team1PlayerTypeCountArray = {0, 0}

local team2PlayerTypeCountArray = {0, 0}

for playerIndex, unwrappedClusterNumberVector in ipairs(assignedClusterNumberVector) do

   local clusterNumber = unwrappedClusterNumberVector[1]

   local player = playerIndexMapping[playerIndex]

  local team1PlayerTypeCount = team1PlayerTypeCountArray[clusterNumber]

  local team2PlayerTypeCount = team2PlayerTypeCountArray[clusterNumber]

  local teamToAssign

  if (team1PlayerTypeCount < team2PlayerTypeCount) then

    team1PlayerTypeCountArray[clusterNumber] = team1PlayerTypeCount + 1

    teamToAssign = 1

  elseif (team2PlayerTypeCount < team1PlayerTypeCount) then

    team2PlayerTypeCountArray[clusterNumber] = team2PlayerTypeCount + 1

    teamToAssign = 2

  else

    

  end

  assignTeam(player, teamToAssign)

end


```


## Resetting Our Team Balancing System

By default, when you reuse the machine learning models from DataPredict, it will interact with the existing model parameters. As such, we need to reset the model parameters by calling the setModelParameters() function and set it to "nil".

```lua

TeamBalancingModel:setModelParameters(nil)

```

That's all for today!
