# Creating Play Time Maximization Model

For this tutorial, we need multiple things to build our model, this includes:

* Neural Network Model

* A Reinforcement Learning Model (Deep Q Learning or Deep SARSA)

* Categorical Policy Quick Setup

## Designing Our Feature Vector And Classes List

Before we start creating our model, we first need to visualize on how we will design our data and what actions the model could take to extend our players' playtime.

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
  "BossSpawnEvent",
  "QuestEvent",
  "ItemSpawnEvent",
  "LimitedTimeQuestEvent",
  "LimitedTimeItemSpawnEvent",

}

```

Also, we would like you to be careful about limited time quest and item spawn events as the model will might learn to give it often. As such, it is important to give the model negative rewards inversely proportional to the duration between the two limited time events.

## Constructing Our Model

Before we start training our model, we first build our model.

### Constructing Our Neural Network

```lua 

local NeuralNetwork = DataPredict.Model.NeuralNetwork.new({maximumNumberOfIterations = 1})

NeuralNetwork:setClassesList(ClassesList)

NeuralNetwork:addLayer(5, true) -- Five features and one bias.

NeuralNetwork:addLayer(#ClassesList, false) -- No bias.

```

### Constructing Our Deep Reinforcement Learning Model

```lua

-- You can use deep Q-Learning here for faster learning. However, for more "safer" model, stick with deep SARSA

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
