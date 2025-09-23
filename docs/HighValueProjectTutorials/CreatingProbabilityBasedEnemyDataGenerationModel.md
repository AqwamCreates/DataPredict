# Creating Probability-Based Enemy Data Generation Model

Hi guys! In this tutorial, we will demonstrate on how to create probability-based enemy data generation model so that the enemies are not too easy or too hard for everyone in PvE modes.

For best results, you must use one class support vector machine.

## Initializing The Probability Model

Before we can produce ourselves a difficulty generation model, we first need to construct a model, which is shown below. Ensure that the kernel function is "RadialBasisFunction".

```lua

local EnemyDataGenerationModel = DataPredict.Models.OneClassSupportVectorMachine.new({maximumnumberOfIterations = 100, kernelFunction = "RadialBasisFunction", beta = 0.9})

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

## Getting The Center Of Clusters

Once you created the feature matrix, you must call model's train() function. This will generate the model parameters.

```lua

EnemyDataGenerationModel:train(playerCombatDataAndEnemyDataMatrix)

```

## Generating The Enemy Data Using The Center Of Clusters

Since we have three clusters, we can expect three rows for our matrix. As such we can process our game logic here.

```lua

for clusterIndex, unwrappedClusterVector in ipairs(ModelParameters) do

  local playerBaseHealth = unwrappedClusterVector[1]
  
  local playerBaseDamage = unwrappedClusterVector[2]
  
  local playerBaseCashAmount = unwrappedClusterVector[3]

  local enemyHealth = playerBaseHealth * 0.5

  local enemyDamage = playerDamage * 0.1

  local enemyCashReward =  playerBaseCashAmount / 3

  spawnEnemy(enemyHealth, enemyDamage, enemyCashReward)

end

```

That's all for today!
