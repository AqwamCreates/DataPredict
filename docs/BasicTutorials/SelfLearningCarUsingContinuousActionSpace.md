# Self-Learning Car Using Continuous Action Space

## Requirements

* Knowledge on how to build neural networks, which can be found [here](UsingNeuralNetworksPart1.md).

## Actions List

```lua

local numberOfActions = 2 -- Steering, Throttle

local actionDimensionSizeArray = {1, numberOfActions}

```

## Environment Feature Vector

### Basic

```lua

local environmentFeatureVector = {

  {1, frontRaycastDistance, backRaycastDistance, leftRaycastDistance, rightRaycastDistance} -- 1 is added at first column for bias, but it is optional.

}

```

### Previous Environment Feature Memory (Optional)

```lua

local memoryEnvironmentFeatureVector = {

  {previousFrontRaycastDistance, previousBackRaycastDistance, previousLeftRaycastDistance, previousRightRaycastDistance}

}

environmentFeatureVector = TensorL:concatenate(environmentFeatureVector, memoryEnvironmentFeatureVector, 2)

```

### Previous Action Memory (Optional)

```lua

environmentFeatureVector = TensorL:concatenate(environmentFeatureVector, meanActionVector, 2)

```

## Setting Up Our Reinforcement Learning Model

There are two ways we can setup our models for our environment:

* Classic Setup (Recommended for experts or people who wants more control over their models)

* Quick Setup (Recommended for beginners)

Below we will show you the difference between the two above. But first, let's define a number of variables and setup the first part of the model.

```lua

local numberOfInputs = #environmentFeatureVector - 1 -- -1 is added to exclude bias from our total environment feature count.

local NeuralNetwork = DataPredict.Models.NeuralNetwork

-- Create the NeuralNetwork models first. We will need two different ones.

local ActorModel = NeuralNetwork.new()

-- Actor NeuralNetwork for controlling actions.

ActorModel:addLayer(numberOfInputs, true, "None") 

ActorModel:addLayer(numberOfActions, false, "None") -- Be careful when choosing the activation function at the final layer. For most use cases, I recommend you to stick with "None" or "Tanh" due to symmetric output values.

-- Critic NeuralNetwork for evaluating actions.

local CriticModel = NeuralNetwork.new()

CriticModel:addLayer(numberOfInputs, true, "None") 

CriticModel:addLayer(1, false, "LeakyReLU") -- Critic only output one value.

local TemporalActorCriticModel = DataPredict.Models.TemporalActorCritic.new({ActorModel = ActorModel, CriticModel = CriticModel}) -- Then create the Temporal Actor-Critic model.

```

## Classic Setup

```lua

while true do

  local previousEnvironmentFeatureVector = initializeEnvironmentFeatureVector() -- We must keep track our previous environment feature vector.

  local previousMeanActionVector = TensorL:createTensor(actionDimensionSizeArray, 0)

  for step = 1, 1000, 1 do

    local currentEnvironmentFeatureVector, reward = fetchEnvironmentFeatureVector(previousEnvironmentFeatureVector, previousMeanActionVector)

    local currentMeanActionVector = TemporalActorCriticModel:predict(currentEnvironmentFeatureVector, true)

    local hasGameEnded = checkIfGameHasEnded(currentEnvironmentFeatureVector)

    local terminalStateValue = (hasGameEnded and 1) or 0

    --[[

      diagonalGaussianUpdate() is called whenever a step is made. The value of zero indicates that the current environment feature vector is not a terminal state.

    --]]

    TemporalActorCriticModel:diagonalGaussianUpdate(previousEnvironmentFeatureVector, previousMeanActionVector, reward, currentEnvironmentFeatureVector, currentMeanActionVector, terminalStateValue)

    previousEnvironmentFeatureVector = currentEnvironmentFeatureVector

    if hasGameEnded then break end

    previousAction = currentAction

  end

 --[[

   episodeUpdate() is used whenever an episode ends. 
   An episode is the total number of steps that determines when the model should stop training.
   The value of one indicates that the current environment feature vector is a terminal state.
 
 --]] 

  TemporalActorCriticModel:episodeUpdate(1)

end

```

As you can see, there are a lot of things that we must track of, but it gives you total freedom on what you want to do with the reinforcement learning models.

## Quick Setup

To reduce the amount of things we need to track, we can use SingleCategoricalPolicy in "QuickSetups" section.

```lua

local TemporalActorCriticQuickSetup = DataPredict.QuickSetups.SingleCategoricalPolicy.new({Model = DeepQLearning})

local previousEnvironmentFeatureVector = initializeEnvironmentFeatureVector() -- We must keep track our previous environment feature vector.

local meanActionVector = TensorL:createTensor(actionDimensionSizeArray, 0)

local reward = 0

while true do

  currentMeanActionVector = TemporalActorCriticQuickSetup:reinforce(environmentFeatureVector, reward)

  environmentFeatureVector, reward = fetchEnvironmentFeatureVector(environmentFeatureVector, meanActionVector)

end

```

As you can see, the SingleCategoricalPolicy compresses a number of codes into reinforce() function.

# Wrapping It All Up

In this tutorial, you have learnt the starting point of the deep reinforcement learning. 

In future lessons, you will learn on how to handle a variety of situations as well as improving the performance of these deep reinforcement learning models.

That's all for today!
