# Creating Reward-Maximization-Based Targeting Model

For this tutorial, we need multiple things to build our model, this includes:

* Neural Network Model

* A Reinforcement Learning Model (Deep Q-Learning or Deep SARSA)

* Categorical Policy Quick Setup

## Designing Our Feature Vector And Classes List

Before we start creating our model, we first need to visualize on how we will design our data and what actions the model could take to extend our players' play time.

### FeatureVector

```lua

-- We have five features with one "bias".

local playerDistanceDifferenceDataVector = {
    {
        1,
        player1PositionDifferenceX,
        player1PositionDifferenceY,
        player1PositionDifferenceZ,
        player2PositionDifferenceX,
        player2PositionDifferenceY,
        player2PositionDifferenceZ,
        player3PositionDifferenceX,
        player3PositionDifferenceY,
        player3PositionDifferenceZ,
    }
}

```

### ClassesList

```lua

local ClassesList = {

  "None",
  "Forward",
  "Backward",
  "Left",
  "Right",
  "Up",
  "Down",
  "Mark",

}

```

If you're concerned about that the model may produce wrong result heavily upon first start up, then you can use a randomized dataset to heavily skew the prediction to the "None" class. Then use this randomized dataset to pretrain the Neural Network before doing any real-time training and prediction. Below, we will show you how it is done.

```lua

local numberOfData = 100

local numberOfPlayers = 3

local numberOfDirections = 3

local numberOfFeatures = (numberOfPlayers * numberOfDirections) + 1 -- Includes one bias.

local randomPlayerDataMatrix = TensorL:createRandomUniformTensor({numberOfData, numberOfFeatures}, -100, 100)

local labelDataMatrix = TensorL:createTensor({numberOfData, 1}, "None")

```

However, this require setting the Neural Network's parameters to these settings temporarily so that it can be biased to "None" at start up as shown below.

```lua

NeuralNetwork.maximumNumberOfIterations = 1000

NeuralNetwork.learningRate = 0.3

```

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

local TargettingModel = DataPredict.QuickSetups.CategoricalPolicy.new()

-- Inserting our Deep Reinforcement Learning Model here.

TargettingModel:setModel(DeepReinforcementLearningModel)

```

## Training And Prediction

Because the way we have designed our Categorical Policy Quick Setup, you can immediately train while producing predictions for your player by calling reinforce() function.

This is because reinforce() function is responsible for producing prediction and perform pre-calculations at the same time as that is required to train our models.

```lua

-- Here, you notice that there is a reward value being inserted here. Generally, when you first call this, the reward value should be zero.

local actionName = PlayTimeMaximizationModel:reinforce(playerDataVector, rewardValue)

```

## Rewarding Our Model

In order to assign the reward to that event is selected, we must first deploy the chosen event and observe if the player stayed for that event.

Below, it shows an example code for this.

```lua

local actionFunctionDictionary = {

  ["None"] = nil,
  ["Forward"] = up,
  ["Backward"] = down,
  ["Left"] = left,
  ["Right"] = right,
  ["Up"] = up,
  ["Down"] = down,
  ["Mark"] = mark,

}

local function run()

    local rewardValue = 0

    local playerDistanceDifferenceDataVector

    local actionName

    local actionFunction

    local heartbeatConnection = Runservice.Heartbeat:Connect(function()

        playerDistanceDifferenceDataVector = getPlayerDistanceDifferenceDataVector(Player)
    
        actionName = PlayTimeMaximizationModel:reinforce(playerLocationDataVector, rewardValue)

        actionFunction = actionFunctionDictionary[eventName]

        if (actionFunction) then actionFunction() end

        -- Calculate the reward based on certain arbitary criteria like how many players it just killed or how fast the players are killed.

        rewardValue = getRewardValue()

    end)

end

```

## Model Parameters Loading 

In here, we will use our model parameters so that it can be used to load out models. There are three cases in here:

1. The player is a first-time player.

2. The player is a returning player.

3. Every player uses the same global model.

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

local DeepReinforcementLearningModel =  TargettingModel:getModel()

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

That's all for today!
