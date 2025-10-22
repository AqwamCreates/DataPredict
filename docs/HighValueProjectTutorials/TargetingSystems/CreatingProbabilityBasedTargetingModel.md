# Creating Probability-Based Targeting Model

Hi guys! In this tutorial, we will demonstrate on how to create probability-based targeting model for your weapon systems.

## Initializing The One Class Support Vector Machine Model

Before we can produce ourselves a targeting model, we first need to construct a model, which is shown below. Ensure that the kernel function is "RadialBasisFunction".

```lua

 -- For this tutorial, we will assume that we want to hit 90% of the players. So, our beta must set to 0.9.

local TargetingModel = DataPredict.Models.OneClassSupportVectorMachine.new({maximumNumberOfIterations = 10, kernelFunction = "RadialBasisFunction", beta = 0.9})

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

local ModelParameters = TargetingModel:getModelParameters()

```

## Interacting With The Players' Location

Currently, we have two case on determining the target location:

* Case 1: Probability

  * Great for performing inaccurate but likely to hit a player. 

  * Can target empty areas to create a probabilistic strike pattern, but is still likely to hit a player near the learned area.

* Case 2: Support Vector Centers

  * Great for maximum number of hits.

### Case 1: Probability

```lua

for i, unwrappedPlayerLocationDataVector in ipairs(playerLocationDataMatrix) do

  local probabilityOfHitting = TargetingModel:predict({unwrappedPlayerLocationDataVector})[1][1]

  --[[

   This is just the tutorial's custom logic where if the model targets a low probability area, the more it will launch missiles there.

   You can put your own game logic here, but we're giving you ideas here.

  --]]

  local probabilityOfMissing = 1 - probabilityOfHitting

  if (probabilityOfMissing < math.random()) then continue end

  local x = unwrappedPlayerLocationDataVector[1]
  
  local y = unwrappedPlayerLocationDataVector[2]
  
  local z = unwrappedPlayerLocationDataVector[3]

  landMissileAt(x, y, z)

end

```

### Case 2: Support Vector Centers

```lua

local ModelParameters = TargetingModel:getModelParameters()

local x = ModelParameters[1][1]
  
local y = ModelParameters[2][1]
  
local z = ModelParameters[3][1]

landMissileAt(x, y, z)

```

## Resetting Our Targeting System

By default, when you reuse the machine learning models from DataPredict, it will interact with the existing model parameters. As such, we need to reset the model parameters by calling the setModelParameters() function and set it to "nil".

```lua

TargetingModel:setModelParameters(nil)

```

That's all for today!
