# Self-Learning Sword-Fighting AIs Using Dicrete Action Space

## Requirements

* Knowledge on how to build neural networks, which can be found [here](UsingNeuralNetworksPart1.md).

## Actions List

```lua

local ActionsList = {"Forward", "Backward", "Left", "Right", "Jump", "Attack", "None"}

```

## Environment Feature Vector

### Basic

```lua

local environmentFeatureVector = {

  {1, health, enemyHealth, enemyDistance, isEnemyLooking} -- 1 is added at first column for bias, but it is optional.

}

```

### Previous Environment Feature Memory (Optional)

```lua

local memoryEnvironmentFeatureVector = {

  {previousHealth, previousEnemyHealth, previousEnemyDistance, previousIsEnemyLooking}

}

environmentFeatureVector = TensorL:concatenate(environmentFeatureVector, memoryEnvironmentFeatureVector, 2)

```

### Previous Action Memory (Optional)

```lua

local memoryActionVector = {

  {didForward, didBackward, didLeft, didRight, didJump, didAttack, didNone} -- 0 and 1 values only, where 1 means that action has been taken previously.

}

environmentFeatureVector = TensorL:concatenate(environmentFeatureVector, memoryActionVector, 2)

```

## Setting Up Our Reinforcement Learning Model

There are two ways we can setup our models for our environment:

* Classic Setup (Recommended for experts or people who wants more control over their models)

* Quick Setup (Recommended for beginners)

Below we will show you the difference between the two above. But first, let's define a number of variables and setup the first part of the model.

```lua

-- Create the NeuralNetwork model first.

local NeuralNetwork = DataPredict.Models.NeuralNetwork.new({ClassesList = ActionsList})

NeuralNetwork:addLayer((#environmentFeatureVector - 1), true, "None") -- -1 is added to exclude bias from our total environment feature count.

NeuralNetwork:addLayer(#ActionList, false, "LeakyReLU")

local DeepQLearning = DataPredict.Models.DeepQLearning.new({Model = NeuralNetwork}) -- Then create the DeepQLearning model.

```

## Classic Setup

```lua

while true do

  local previousEnvironmentFeatureVector = initializeEnvironmentFeatureVector() -- We must keep track our previous environment feature vector.

  local previousAction = "None"

  for step = 1, 1000, 1 do

    local currentEnvironmentFeatureVector, reward = fetchEnvironmentFeatureVector(previousEnvironmentFeatureVector, previousAction)

    local currentAction = DeepQLearning:predict(currentEnvironmentFeatureVector)[1][1]

    local hasGameEnded = checkIfGameHasEnded(currentEnvironmentFeatureVector)

    local terminalStateValue = (hasGameEnded and 1) or 0

    --[[

      categoricalUpdate() is called whenever a step is made. The value of zero indicates that the current environment feature vector is not a terminal state.

    --]]

    DeepQLearning:categoricalUpdate(previousEnvironmentFeatureVector, previousAction, reward, currentEnvironmentFeatureVector, currentAction, terminalStateValue)

    previousEnvironmentFeatureVector = currentEnvironmentFeatureVector

    previousAction = currentAction

    if hasGameEnded then break end

  end

  --[[

    episodeUpdate() is used whenever an episode ends.
    An episode is the total number of steps that determines when the model should stop training.
    The value of one indicates that the current environment feature vector is a terminal state.

  --]] 

  QLearningNeuralNetwork:episodeUpdate(1)

end

```

As you can see, there are a lot of things that we must track of, but it gives you total freedom on what you want to do with the reinforcement learning models.

## Quick Setup

To reduce the amount of things we need to track, we can use SingleCategoricalPolicy in "QuickSetups" section.

```lua

local DeepQLearningQuickSetup = DataPredict.QuickSetups.SingleCategoricalPolicy.new({Model = DeepQLearning})

local environmentFeatureVector = initializeEnvironmentFeatureVector() -- We must keep track our previous environment feature vector.

local action = "None"

local reward = 0

while true do

  action = DeepQLearningQuickSetup:reinforce(environmentFeatureVector, reward)

  environmentFeatureVector, reward = fetchEnvironmentFeatureVector(environmentFeatureVector, action)

end

```

As you can see, the SingleCategoricalPolicy compresses a number of codes into reinforce() function.

# Wrapping It All Up

In this tutorial, you have learnt the starting point of the deep reinforcement learning. 

In future lessons, you will learn on how to handle a variety of situations as well as improving the performance of these deep reinforcement learning models.

That's all for today!
