# Creating Play Time Maximization Model

For this tutorial, we need multiple things to build our model, this includes:

* Neural Network Model

* A Reinforcement Learning Model (Deep Q Learning or Deep SARSA)

* Categorical Policy Quick Setup

## Designing Our Feature Vector And Classes List

Before we start creating our model, we first need to visualize on how we will design our data and what actions the model could take to extend our players' play time.

### FeatureVector

```lua

-- We have five features with one "bias".

local initialPlayerDataVector = {
    {
        1,
        numberOfCurrencyAmount,
        numberOfItemsAmount,
        timePlayedInCurrentSession,
        timePlayedInAllSessions,
        healthAmount
    }
}

```

### ClassesList

```lua

local ClassesList = {

  "NoEvent",
  "ResourceMultiplierEvent",
  "QuestEvent",
  "ItemSpawnEvent",
  "BossSpawnEvent",
  "LimitedTimeQuestEvent",
  "LimitedTimeItemSpawnEvent",
  "LimitedTimeBossSpawnEvent",

}

```

Also, we would like you to be careful about limited time quest and item spawn events as the model will might learn to give it often. As such, it is important to give the model negative rewards inversely proportional to the duration between the two limited time events.

## Constructing Our Model

Before we start training our model, we first need to build our model. We have split this to multiple subsections to make it easy to follow through.

### Constructing Our Neural Network

```lua 

local NeuralNetwork = DataPredict.Model.NeuralNetwork.new({maximumNumberOfIterations = 1})

NeuralNetwork:setClassesList(ClassesList)

NeuralNetwork:addLayer(5, true) -- Five features and one bias.

NeuralNetwork:addLayer(#ClassesList, false) -- No bias.

```

### Constructing Our Deep Reinforcement Learning Model

```lua

-- You can use deep Q-Learning here for faster learning. However, for more "safer" model, stick with deep SARSA.

local DeepReinforcementLearningModel = DataPredict.Model.DeepStateActionRewardStateAction.new()

-- Inserting our Neural Network here.

DeepReinforcementLearningModel:setModel(NeuralNetwork)

```

### Constructing Our Categorical Policy Quick Setup Model

This part makes it easier for us to set up our model, but it is not strictly necessary. However, I do recommend you to use them as they contain built-in functions for handing training and predictions.

```lua

local PlayTimeMaximizationModel = DataPredict.QuickSetups.CategoricalPolicy.new()

-- Inserting our Deep Reinforcement Learning Model here.

PlayTimeMaximizationModel:setModel(DeepReinforcementLearningModel)

```

## Training And Prediction

Because the way we have designed our Categorical Policy Quick Setup, you can immediately train while producing predictions for your player by calling reinforce() function.

This is because reinforce() function is responsible for producing prediction and perform pre-calculations at the same time as that is required to train our models.

```lua

-- Here, you notice that there is a reward value being inserted here. Generally, when you first call this, the reward value should be zero.

local action = PlayTimeMaximizationModel:reinforce(initialPlayerDataVector, rewardValue)

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
