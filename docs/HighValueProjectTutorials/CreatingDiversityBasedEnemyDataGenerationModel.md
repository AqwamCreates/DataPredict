# Creating Diversity-Based Enemy Data Generation Model

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

Multiple cases can be done here.

* Case 1: Binary Generation.

  * For a given set of generated enemy data values, the model determines the probability that the player will interact with it. This is then used to spawn or reject the enemy with the generated data values.

* Case 2: Weighted Generation

  * For a given set of generated enemy data values, the model outputs a probability that can be used to modify the generated enemy data.

  * General formula: generatedValue = bestValue * probabilityToInteractFromSupportVectorMachine. Hence, bestValue = generatedValue / probabilityToInteractFromSupportVectorMachine.
 
  * Once bestValue is calculated, spawn an enemy with this best value data.

But first, let initialize an array so that we can control how many enemies we should generate.

```lua

local activeEnemyDataArray = {}

local maximumNumberOfEnemies = 10

```

Optionally, we can also generate enemy data vector based on the model parameters.

```lua

local function generateEnemyDataVector()

 local ModelParameters = EnemyDataGenerationModel:getModelParameters()

 local enemyMaximumHealth = ModelParameters[1][1]

 local enemyMaximumDamage = ModelParameters[2][1]

 local enemyCashAmount = ModelParameters[3][1]

 local enemyMaximumHealthRandomNoise = math.random() - math.random()

 local enemyMaximumDamageRandomNoise = math.random() - math.random()

 local enemyCashAmountRandomNoise = math.random() - math.random()

 return {{enemyMaximumHealth * enemyMaximumHealthRandomNoise, enemyMaximumDamage * enemyMaximumHealthRandomNoise, enemyCashAmount * enemyCashAmountRandomNoise}}

end

```

### Case 1: Binary Generation.

```lua

local playerCombatDataVector

local enemyDataVector

local playerCombatDataAndEnemyDataVector

local probabilityForPlayerToInteract

local isAcceptable = false

while true do

  if (#activeEnemyDataArray > maximumNumberOfEnemies) then continue end

  repeat

    playerCombatDataVector = getPlayerDataVector()

    enemyDataVector = generateEnemyDataVector()

    playerCombatDataAndEnemyDataVector = TensorL:concatenate(playerCombatDataVector, enemyDataVector, 2)

    probabilityForPlayerToInteract = EnemyDataGenerationModel:predict(playerCombatDataAndEnemyDataVector)[1][1]

    isAcceptable = (probabilityForPlayerToInteract >= 0.5)

  until isAcceptable

  summonEnemy(enemyDataVector)

end

```

### Case 2: Weighted Generation.

```lua

local playerCombatDataVector

local enemyDataVector

local playerCombatDataAndEnemyDataVector

local probabilityForPlayerToInteract

while true do

  if (#activeEnemyDataArray > maximumNumberOfEnemies) then continue end

 playerCombatDataVector = getPlayerDataVector()

 enemyDataVector = generateEnemyDataVector()

 playerCombatDataAndEnemyDataVector = TensorL:concatenate(playerCombatDataVector, enemyDataVector, 2)

 probabilityForPlayerToInteract = EnemyDataGenerationModel:predict(playerCombatDataAndEnemyDataVector)[1][1]

 enemyDataVector = TensorL:divide(enemyDataVector, probabilityForPlayerToInteract)

 summonEnemy(enemyDataVector)

end

```

## Upon Player Interaction With Enemy.

```lua

--[[

You can keep all the data or periodically clear it upon model training.

I recommend the latter because it makes sure we don't include old data that might not be relevant to the current session.

Additionally, using the whole data is computationally expensive and may impact players' gameplay experience.

--]]

local playerCombatDataAndEnemyDataMatrix = {}

local function onEnemyKilled(Enemy, Player)

  local playerCombatDataVector = getPlayerCombatDataVector(Player)

  local enemyDataVector = getEnemyDataVector(Enemy)

  local playerCombatDataAndEnemyDataVector = TensorL:concatenate(playerCombatDataVector, enemyDataVector, 2)

  table.insert(playerCombatDataAndEnemyDataMatrix, playerCombatDataAndEnemyDataVector[1])

  removeEnemyDataFromActiveEnemyDataArray(Enemy)

end

```

That's all for today!
