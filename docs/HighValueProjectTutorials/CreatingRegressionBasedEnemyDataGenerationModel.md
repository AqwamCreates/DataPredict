# Creating Regression-Based Enemy Data Generation Model

Hi guys! In this tutorial, we will demonstrate on how to create regression-based enemy data generation model so that the enemies are personalized to each players.

For best results, please use:

* Neural Networks

* Regression Models

## Initializing The Model

Before we can produce ourselves a difficulty generation model, we first need to construct a model, which is shown below. Ensure to add rows of 1 to add "bias".

```lua

local EnemyDataGenerationModel = DataPredict.Models.NeuralNetwork.new() -- Set the maximumNumberOfIterations to 1 if you want the incremental version.

EnemyDataGenerationModel:addLayer(2, true) -- This is specific to neural networks only.

EnemyDataGenerationModel:addLayer(3, false)

```

## Collecting The Players' Combat Data

Before we can train our models, we first need all the players' combat data and defeated enemy's combat data into separate matrices. Ensure that the rows matches the player-enemy pairs.

```lua

local playerCombatDataMatrix = {

  {1, player1MaximumHealth, player1MaximumDamage},
  {1, player2MaximumHealth, player2MaximumDamage},
  {1, player3MaximumHealth, player3MaximumDamage},

}

local defeatedEnemyDataMatrix = {

  {enemy1MaximumHealth, enemy1MaximumDamage, enemy1CashAmount},
  {enemy2MaximumHealth, enemy2MaximumDamage, enemy2CashAmount},
  {enemy3MaximumHealth, enemy3MaximumDamage, enemy3CashAmount},

}

```

## Training The Model

Once you collected the players' combat data, you must call model's train() function. This will generate the model parameters.

```lua

EnemyDataGenerationModel:train(playerCombatDataMatrix, defeatedEnemyDataMatrix)

```

## Generating The Difficulty

```lua

local generatedEnemyDataVector = EnemyDataGenerationModel:predict(playerCombatDataMVector, true) -- Since neural network defaults to classification, you must set returnOriginalOutput to "true" so that it becomes a regression model.

local unwrappedGeneratedEnemyDataVector = generatedEnemyDataVector[1]

local generatedEnemyMaximumHealth = unwrappedGeneratedEnemyDataVector[1]

local generatedEnemyMaximumDamage = unwrappedGeneratedEnemyDataVector[2]

local generatedEnemyCashAmount = unwrappedGeneratedEnemyDataVector[3]

```

## Model Parameters Loading 

In here, we will use our model parameters so that it can be used to load out models. There are three cases in here:

1. The player is a first-time player.

2. The player is a returning player.

3. Every player uses the same global model.

### Case 1: The Player Is A First-Time Player

Under this case, this is a new player that plays the game for the first time. In this case, we do not know how this player would act.

We have a multiple way to handle this issue:

* We create a "global" model that trains from every player, and then make a deep copy of the model parameters and load it into our models.

* We take from other players' existing model parameters and load it into our models.

### Case 2: The Player Is A Returning Player

Under this case, you can continue using the existing model parameters that was saved in Roblox's Datastores.

```lua

-- Notice that we must get it from the Neural Network model.

ModelParameters = EnemyDataGenerationModel:getModelParameters()

-- Notice that we must set it to the Neural Network model too.

EnemyDataGenerationModel:setModelParameters(ModelParameters)

```

### Case 3: Every Player Uses The Same Global Model

Under this case, the procedure is the same to case 2 except that you need to:

* Load model parameters upon server start.

* Perform auto-save with the optional ability of merging with saved model parameters from other servers.

That's all for today!
