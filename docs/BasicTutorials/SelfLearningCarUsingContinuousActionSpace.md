# Self-Learning Car Using Continuous Action Space

## Requirements

* Knowledge on how to build neural networks, which can be found [here](UsingNeuralNetworksPart1.md).

## Actions Vectors

### General Action Information

```lua

local numberOfActions = 2 -- Steering Angle, Speed

local actionDimensionSizeArray = {1, numberOfActions}

```

### Action Standard Deviation Information

```lua

--[[

  Maximum steering angle for one side is 90 degrees. We can consider this as our variance.

  You can decrease this if you want to reduce risky exploration.

--]]

local steerAngleStandardDeviation = math.sqrt(90)

--[[

  Humans often increase and decrease by values of ten for kilometers per hour. We can consider this as our variance.

  You can decrease this if you want to reduce risky exploration.

--]]

local speedStandardDeviation = math.sqrt(10)

local standardDeviationActionVector = { -- This control how far the values can go from the center / mean.

  {steerAngleStandardDeviation, speedStandardDeviation}

}

```

## Environment Feature Vector

### Basic

```lua

--[[

  1 is added at first column for bias, but it is optional.

  Additionally, continous action space deep reinforcement learning models are extremely sensitive to rewards. Hence we use inverse distance for our inputs to stabilize our model's training.

--]]

local environmentFeatureVector = {

  {1, inverseFrontRaycastDistance, inverseBackRaycastDistance, inverseLeftRaycastDistance, inverseRightRaycastDistance} 

}

```

### Previous Environment Feature Memory (Optional)

```lua

local memoryEnvironmentFeatureVector = {

  {previousInverseFrontRaycastDistance, previousInverseBackRaycastDistance, previousInverseLeftRaycastDistance, previousInverseRightRaycastDistance}

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

local TemporalDifferenceActorCriticModel = DataPredict.Models.TemporalDifferenceActorCritic.new({ActorModel = ActorModel, CriticModel = CriticModel}) -- Then create the TemporalDifference Actor-Critic model.

```

## Classic Setup

```lua

while true do

  local previousEnvironmentFeatureVector = initializeEnvironmentFeatureVector() -- We must keep track our previous environment feature vector.

  local previousActionMeanVector = TensorL:createTensor(actionDimensionSizeArray, 0)

  local previousActionNoiseVector

  for step = 1, 1000, 1 do

    local currentEnvironmentFeatureVector, reward = fetchEnvironmentFeatureVector(previousEnvironmentFeatureVector, previousMeanActionVector)

    local currentActionMeanVector = TemporalDifferenceActorCriticModel:predict(currentEnvironmentFeatureVector, true)

    local previousActionNoiseVector = TensorL:createRandomNormalTensor(actionDimensionSizeArray)

    local hasGameEnded = checkIfGameHasEnded(currentEnvironmentFeatureVector)

    local terminalStateValue = (hasGameEnded and 1) or 0

    --[[

      diagonalGaussianUpdate() is called whenever a step is made. The value of zero indicates that the current environment feature vector is not a terminal state.

    --]]

    TemporalDifferenceActorCriticModel:diagonalGaussianUpdate(previousFeatureVector, previousActionMeanVector, previousActionStandardDeviationVector, previousActionNoiseVector, rewardValue, currentFeatureVector, currentActionMeanVector, terminalStateValue)

    previousEnvironmentFeatureVector = currentEnvironmentFeatureVector

    previousActionMeanVector = currentActionMeanVector

    if hasGameEnded then break end

    previousAction = currentAction

  end

 --[[

   episodeUpdate() is used whenever an episode ends. 
   An episode is the total number of steps that determines when the model should stop training.
   The value of one indicates that the current environment feature vector is a terminal state.
 
 --]] 

  TemporalDifferenceActorCriticModel:episodeUpdate(1)

end

```

As you can see, there are a lot of things that we must track of, but it gives you total freedom on what you want to do with the reinforcement learning models.

## Quick Setup

To reduce the amount of things we need to track, we can use SingleCategoricalPolicy in "QuickSetups" section.

```lua

local TemporalDifferenceActorCriticQuickSetup = DataPredict.QuickSetups.SingleDiagonalGaussianPolicy.new({Model = TemporalDifferenceActorCriticModel, standardDeviationActionVector = standardDeviationActionVector})

local previousEnvironmentFeatureVector = initializeEnvironmentFeatureVector() -- We must keep track our previous environment feature vector.

local actionMeanVector = TensorL:createTensor(actionDimensionSizeArray, 0)

local reward = 0

while true do

  actionMeanVector = TemporalDifferenceActorCriticQuickSetup:reinforce(environmentFeatureVector, reward)

  environmentFeatureVector, reward = fetchEnvironmentFeatureVector(environmentFeatureVector, actionMeanVector)

end

```

As you can see, the SingleDiagonalGaussianPolicy compresses a number of codes into reinforce() function.

# Wrapping It All Up

In this tutorial, you have learnt the starting point of the deep reinforcement learning. 

In future lessons, you will learn on how to handle a variety of situations as well as improving the performance of these deep reinforcement learning models.

That's all for today!
