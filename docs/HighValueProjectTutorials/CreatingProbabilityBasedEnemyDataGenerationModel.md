# Creating Probability-Based Enemy Data Generation Model

Hi guys! In this tutorial, we will demonstrate on how to create probability-based enemy data generation model so that the enemies are not too easy or too hard for everyone in PvE modes.

For best results, you must use one class support vector machine.

## Initializing The Probability Model

Before we can produce ourselves a difficulty generation model, we first need to construct a model, which is shown below. Ensure that the kernel function is "RadialBasisFunction".

```lua

 -- For this tutorial, we will assume that the player intentionally killed 90% of the enemies.

local EnemyDataGenerationModel = DataPredict.Models.OneClassSupportVectorMachine.new({maximumNumberOfIterations = 100, kernelFunction = "RadialBasisFunction", beta = 0.9})

```

## Designing The Feature Matrix

Before we can train and generate our models, we first need to design our featureMatrix.

```lua

-- Techincally, the player combat data information is not quite necessary unless these values changes a lot or you're using it as part of enemy data generation.

local playerCombatDataMatrix = {

  {player1MaximumHealth, player1MaximumDamage, player1CashAmount},
  {player2MaximumHealth, player2MaximumDamage, player2CashAmount},
  {player3MaximumHealth, player3MaximumDamage, player3CashAmount},

}

local enemyDataMatrix = {

  {enemy1MaximumHealth, enemy1MaximumDamage, enemy1CashAmount},
  {enemy2MaximumHealth, enemy2MaximumDamage, enemy2CashAmount},
  {enemy3MaximumHealth, enemy3MaximumDamage, enemy3CashAmount},

}

local playerCombatDataAndEnemyDataMatrix = TensorL:concatenate(playerCombatDataMatrix, enemyDataMatrix, 2)

```

## Training Our Models

Once you created the feature matrix, you must call model's train() function. This will generate the model parameters.

```lua

EnemyDataGenerationModel:train(playerCombatDataAndEnemyDataMatrix)

```

## Generating The Enemy Data

```lua

local activeEnemyDataArray = {}

--[[

You can keep all the data or periodically clear it upon model training.

I recommend the latter because it makes sure we don't include old data that might not be relevant to the current session.

Additionally, using the whole data is computationally expensive and may impact players' gameplay experience.

--]]

local playerCombatDataAndEnemyDataMatrix = {}

local function onEnemyKilled(Enemy, Player)

  local playerCombatData = getPlayerCombatData(playerCombatData)

  local enemyData = getEnemyData(enemyData)

  local playerCombatDataAndEnemyDataMatrix = TensorL:concatenate(playerCombatDataMatrix, enemyDataMatrix, 2)

end

```

That's all for today!
