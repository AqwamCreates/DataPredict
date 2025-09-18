# Creating Reward-Maximization-Based Regression Recommendation Model

Hello guys! Today, I will be showing you on how to create a reward-maximization-based model that could predict the likelihood that the player will buy the item.

Currently, you need these to produce the model:

* A neural network model

* A reinforcement learning model (Deep-Q-Learning or Deep SARSA)

* A player data that is stored in matrix

* An item data that is stored in matrix

## Designing Our Feature Vector

Before we start creating our model, we first need to visualize on how we will design our data to increase the likelihood of players purchasing an item.

```lua

-- We're just adding 1 here to add "bias".

local playerDataVector = {
    {
        1,
        numberOfCurrencyAmount,
        numberOfItemsAmount,
        timePlayedInCurrentSession,
        timePlayedInAllSessions,
        currentHealthAmount,
        currentDamageAmount
    }
}

local itemDataVector = {
    {
        costAmount,
        rarityValue,
    }
}

```

## Constructing Our Model

Before we start training our model, we first need to build our model. We have split this to multiple subsections to make it easy to follow through.

### Constructing Our Neural Network

```lua 

local ActorNeuralNetwork = DataPredict.Model.NeuralNetwork.new({maximumNumberOfIterations = 1})

ActorNeuralNetwork:addLayer(8, true) -- Six player data features, two item data features and one bias.

ActorNeuralNetwork:addLayer(3, false) -- Three enemy features and no bias.

local CriticNeuralNetwork = DataPredict.Model.NeuralNetwork.new({maximumNumberOfIterations = 1})

CriticNeuralNetwork:addLayer(8, true) -- Six player data features, two item data features and one bias.

CriticNeuralNetwork:addLayer(1, false) -- Critic only outputs 1 value.

```

### Constructing Our Deep Reinforcement Learning Model

```lua

-- You can use deep Q-Learning here for faster learning. However, for more "safer" model, stick with deep SARSA.

local DeepReinforcementLearningModel = DataPredict.Model.SoftActorCritic.new()

-- Inserting our actor and critic Neural Networks here.

DeepReinforcementLearningModel:setActorModel(ActorNeuralNetwork)

DeepReinforcementLearningModel:setCriticModel(CriticNeuralNetwork)

```

### Constructing Our Diagonal Gaussian Policy Quick Setup Model

This part makes it easier for us to set up our model, but it is not strictly necessary. However, I do recommend you to use them as they contain built-in functions for handing training and predictions.

```lua

--[[

The vector below controls how far the enemy feature value should be generated.

Let's say our model decides to make 100 maximum health as our base value for enemy. 

A standard deviation of 10 would make the model generate an enemy with the maximum health between 90 and 110.

--]]

local actionStandardDeviationVector = {

    {probabilityStandardDeviation}

}

-- Next, we'll insert actionStandardDeviationVector to our quick setup constructor.

local RecommendationModel = DataPredict.QuickSetups.DiagonalGaussianPolicy.new({actionStandardDeviationVector = actionStandardDeviationVector})

-- Inserting our Deep Reinforcement Learning Model here.

RecommendationModel:setModel(DeepReinforcementLearningModel)

```

## Prediction And Training

In here, let's assume that we have a shop GUI that the player can interact with. Since the prediction and training are closely related in terms of code, I will be splitting the process in different subsections.

### Upon Player Opening Shop GUI

The code shown below demonstrate on how to generate the recommendation by the time the player opens the GUI.

```lua

local function showRecommendations(itemName, itemDataVector, rewardValue, previousActionMeanValue)

    local currentPlayerData = getPlayerDataVector()

    local playerItemDataPairVector = TensorL:concatenate(playerDataVector, itemDataVector, reward)

     -- Forces the model's action to the selected one to make sure the model updates properly.

    if (previousActionMeanValue) then RecommendationModel.previousActionMeanVector = {{previousActionMeanValue}} end

    local action = RecommendationModel:reinforce(playerItemDataPairVector, rewardValue)

    if (action == "Recommend") then

        recommendItem(itemName)

    else

        recommendARandomItemExceptFor(itemName)

    end

end

local function onShopGUIOpen()

    local randomItemName, randomDataVector = getRandomItem()

    local rewardValue = 0

   showRecommendations(randomItemName, randomDataVector, rewardValue)

end

```

### Upon Item Purchase

```lua

local function onItemPurchase(itemName, itemDataVector)

     local rewardValue = 50

    showRecommendations(itemName, itemDataVector, rewardValue, 1)

end

```

### Upon Player Closing Shop GUI

```lua

local function onShopGUIClose(lastShownItemName, lastItemDataVector)

    local rewardValue = -50

   showRecommendations(lastShownItemName, lastItemDataVector, rewardValue, -1)

end

```

This should give you a model that predicts a rough estimate on what they will likely to buy.

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

local DeepReinforcementLearningModel = EnemyDataGenerationModel:getModel()

local ActorNeuralNetwork = DeepReinforcementLearningModel:getActorModel()

local CriticNeuralNetwork = DeepReinforcementLearningModel:getCriticModel()

-- Notice that we must get it from the actor and critic Neural Network models.

ActorModelParameters = ActorNeuralNetwork:getModelParameters()

CriticModelParameters = CriticNeuralNetwork:getModelParameters()

-- Notice that we must set it to the actor and critic Neural Network models too.

ActorNeuralNetwork:setModelParameters(ActorModelParameters)

CriticNeuralNetwork:setModelParameters(CriticModelParameters)

```

### Case 3: Every Player Uses The Same Global Model

Under this case, the procedure is the same to case 2 except that you need to:

* Load model parameters upon server start.

* Perform auto-save with the optional ability of merging with saved model parameters from other servers.

## Conclusion

This tutorial showed you on how to create item recommendation model that allows you to increase the likelihood the player will purchase an item. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
