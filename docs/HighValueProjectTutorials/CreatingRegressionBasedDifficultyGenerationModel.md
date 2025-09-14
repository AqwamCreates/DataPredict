# Creating Regression-Based Difficulty Generation Model

Hi guys! In this tutorial, we will demonstrate on how to create regression-based difficulty generation model so that the enemies are personalized to each players.

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

local generatedEnemyMaximumHealth = generatedEnemyCombatDataVector[1][1]

local generatedEnemyMaximumDamage = generatedEnemyCombatDataVector[1][2]

local generatedEnemyCashAmount = generatedEnemyCombatDataVector[1][3]

```

That's all for today!
