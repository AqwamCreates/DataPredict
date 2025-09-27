# Creating Conditional-Diversity-Based Enemy Data Generation Model

Hi guys! In this tutorial, we will demonstrate on how to create diversity-based enemy data generation model so that the enemies are not too easy or too hard for everyone in PvE modes.

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

local noiseMatrix = TensorL:createRandomUniformTensor({3, 1}) -- Single point of variation.

```

## Constructing Our Model

Before we start training our model, we first need to build our model. We have split this to multiple subsections to make it easy to follow through.

### Constructing Our Neural Network

```lua 

local GeneratorNeuralNetwork = DataPredict.Model.NeuralNetwork.new({maximumNumberOfIterations = 1})

GeneratorNeuralNetwork:addLayer(4, true) -- Three player data features, one noise feature and one bias.

GeneratorNeuralNetwork:addLayer(3, false) -- We're outputing three enemy data features and is without bias.

local DiscriminatorNeuralNetwork = DataPredict.Model.NeuralNetwork.new({maximumNumberOfIterations = 1})

DiscriminatorNeuralNetwork:addLayer(6, true) -- Three player data features, three enemy features and one bias.

DiscriminatorNeuralNetwork:addLayer(1, false) -- Discriminator only outputs 1 value.

```

### Constructing Our Deep Reinforcement Learning Model

```lua

-- You can use CGAN here. However, for more "stable" model, stick with CWGAN.

local EnemyDataGenerationModel = DataPredict.Model.ConditionalWassersteinGenerativeAdversarialNetwork.new()

-- Inserting our generator and discriminator Neural Networks here.

EnemyDataGenerationModel:setGeneratorModel(GeneratorNeuralNetwork)

EnemyDataGenerationModel:setDiscriminatorModel(DiscriminatorNeuralNetwork)

```

## Training Our Models

Once you created the feature matrix, you must call model's train() function. This will generate the model parameters.

```lua

EnemyDataGenerationModel:train(enemyDataMatrix, noiseDataMatrix, playerCombatDataMatrix)

```

## Generating The Enemy Data

Multiple cases can be done here.

* Case 1: Binary Generation.

  * For a given set of generated enemy data values, the model determines the probability that the player will interact with it. This is then used to spawn or reject the enemy with the generated data values.

* Case 2: Weighted Generation

  * For a given set of generated enemy data values, the model outputs a probability that can be used to modify the generated enemy data.

  * General formula: generatedValue = bestValue * probabilityToInteract. Hence, bestValue = generatedValue / probabilityToInteract.
 
  * Once bestValue is calculated, spawn an enemy with this best value data.

But first, let initialize an array so that we can control how many enemies we should generate.

```lua

local activeEnemyDataArray = {}

local maximumNumberOfEnemies = 10

```

### Case 1: Binary Generation

```lua

local noiseVector

local playerCombatDataVector

local enemyDataVector

local playerCombatDataAndEnemyDataVector

local probabilityForPlayerToInteract

local isAcceptable = false

while true do

  if (#activeEnemyDataArray > maximumNumberOfEnemies) then continue end

  repeat

    noiseVector = {{math.random()}}

    playerCombatDataVector = getPlayerDataVector()

    enemyDataVector = EnemyDataGenerationModel:generate(noiseVector, playerCombatDataVector)

    probabilityForPlayerToInteract = EnemyDataGenerationModel:evaluate(enemyDataVector)[1][1]

    isAcceptable = (probabilityForPlayerToInteract >= 0.5)

  until isAcceptable

  summonEnemy(enemyDataVector)

end

```

### Case 2: Weighted Generation

```lua

local noiseVector

local playerCombatDataVector

local enemyDataVector

local playerCombatDataAndEnemyDataVector

local probabilityForPlayerToInteract

while true do

  if (#activeEnemyDataArray > maximumNumberOfEnemies) then continue end

  noiseVector = {{math.random()}}

  playerCombatDataVector = getPlayerDataVector()

  enemyDataVector = EnemyDataGenerationModel:generate(noiseVector, playerCombatDataVector)

  probabilityForPlayerToInteract = EnemyDataGenerationModel:evaluate(enemyDataVector)[1][1]

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

local playerCombatDataMatrix = {}

local playerEnemyDataMatrix = {}

local function onEnemyKilled(Enemy, Player)

  local playerCombatDataVector = getPlayerCombatDataVector(Player)

  local enemyDataVector = getEnemyDataVector(Enemy)

  table.insert(playerCombatDataMatrix, playerCombatDataAndEnemyDataVector[1])

  table.insert(playerEnemyDataMatrix, enemyDataVector[1])

  removeEnemyDataFromActiveEnemyDataArray(Enemy)

end

```

That's all for today!
