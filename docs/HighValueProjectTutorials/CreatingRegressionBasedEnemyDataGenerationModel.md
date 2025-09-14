# Creating Regression-Based Enemy Data Generation Model

Hi guys! In this tutorial, we will demonstrate on how to create regression-based enemy data generation model so that the enemies are personalized to each players.

For best results, please use:

* Neural Networks

* Regression Models

## Initializing The Model

Before we can produce ourselves a difficulty generation model, we first need to construct a model, which is shown below. Ensure to add rows of 1 to add "bias".

```lua

local DifficultyGenerationModel = DataPredict.Models.NeuralNetwork.new() -- Set the maximumNumberOfIterations to 1 if you want the incremental version.

DifficultyGenerationModel:addLayer(2, true) -- This is specific to neural networks only.

DifficultyGenerationModel:addLayer(3, false)

```

## Collecting The Players' Combat Data

Before we can train our models, we first need all the players' combat data and defeated enemy's combat data into separate matrices. Ensure that the rows matches the player-enemy pairs.

```lua

local playerCombatDataMatrix = {

  {1, player1MaximumHealth, player1MaximumDamage},
  {1, player2MaximumHealth, player2MaximumDamage},
  {1, player3MaximumHealth, player3MaximumDamage},

}

local defeatedEnemyCombatDataMatrix = {

  {1, enemy1MaximumHealth, enemy1MaximumDamage, enemy1CashAmount},
  {1, enemy2MaximumHealth, enemy2MaximumDamage, enemy2CashAmount},
  {1, enemy3MaximumHealth, enemy3MaximumDamage, enemy3CashAmount},

}

```

## Training The Model

Once you collected the players' combat data, you must call model's train() function. This will generate the model parameters.

```lua

DifficultyGenerationModel:train(playerCombatDataMatrix, defeatedEnemyCombatDataMatrix)

```

## Generating The Difficulty

```lua

local generatedEnemyCombatDataVector = DifficultyGenerationModel:predict(playerCombatDataMVector, true) -- Since neural network defaults to classification, you must set returnOriginalOutput to "true" so that it becomes a regression model.

local unwrappedGeneratedEnemyCombatDataVector = generatedEnemyCombatDataVector[1]

local generatedEnemyMaximumHealth = unwrappedGeneratedEnemyCombatDataVector[1]

local generatedEnemyMaximumDamage = unwrappedGeneratedEnemyCombatDataVector[2]

local generatedEnemyCashAmount = unwrappedGeneratedEnemyCombatDataVector[3]

```

## Model Parameters Loading 

In here, we will use our model parameters so that it can be used to load out models. There are two cases in here:

1. The player is a first-time player.

2. The player is a returning player.

### Case 1: The Player Is A First-Time Player

Under this case, this is a new player that plays the game for the first time. In this case, we do not know how this player would act.

We have a multiple way to handle this issue:

* We create a "global" model that trains from every player, and then make a deep copy of the model parameters and load it into our models.

* We take from other players' existing model parameters and load it into our models.

### Case 2: The Player Is A Returning Player

Under this case, you can continue using the existing model parameters that was saved in Roblox's Datastores.

```lua

--[[ 

We first need to get our Neural Network model. If you only kept the quick setup and discarded the rest, don't worry!

We can just do getModel() twice to get our Neural Network model.

--]]

local DeepReinforcementLearningModel =  PlayTimeMaximizationModel:getModel()

local NeuralNetwork = DeepReinforcementLearningModel:getModel()

-- Notice that we must get it from the Neural Network model.

ModelParameters = NeuralNetwork:getModelParameters()

-- Notice that we must set it to the Neural Network model too.

NeuralNetwork:setModelParameters(ModelParameters)

```

### Case 3: Every Player Uses The Same Global Model

Under this case, the procedure is the same to case 2 except that you need to:

* Load model parameters upon server start.

* Perform auto-save with the optional ability of merging with saved model parameters from other servers.

That's all for today!
