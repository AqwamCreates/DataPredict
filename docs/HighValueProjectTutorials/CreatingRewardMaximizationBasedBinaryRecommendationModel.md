# Creating Reward-Maximization-Based Binary Recommendation Model

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
        durationBetweenFirstServerJoinAndThisItemPurchase,
    }
}

```

## Constructing Our Model

Before we start training our model, we first need to build our model.

```lua

local DataPredict = require(DataPredict)

-- For single data point purposes, set the maximumNumberOfIterations to 1 to avoid overfitting. Additionally, the more number of maximumNumberOfIterations you have, the lower the learningRate it should be to avoid "inf" and "nan" issues.

local NeuralNetwork = DataPredict.Models.NeuralNetwork.new({maximumNumberOfIterations = 1, learningRate = 0.3})

NeuralNetwork:setClassesList({"Recommend", "DoNotRecommend"})

NeuralNetwork:addLayer(6, true) -- Six features and one bias.

NeuralNetwork:addLayer(#ClassesList, false) -- No bias.

```

## Prediction And Training

In here, let's assume that we have a shop GUI that the player can interact with. Since the prediction and training are closely related in terms of code, I will be splitting the process in different subsections.

### Upon Player Opening Shop GUI

The code shown below demonstrate on how to generate the recommendation by the time the player opens the GUI.

```lua

local itemToShowArray = {}

local itemDataMatrix = {}

local hasPlayerPurchasedTheItemVector -- We will reserve this for now for readibility

local currentPlayerData = getPlayerDataVector()

for itemName, itemDataVector in pairs(itemDictionary)

    local playerItemDataPairVector = TensorL:concatenate(playerDataVector, itemDataVector, 2)

    local generatedLabelVector = RecommendationModel:predict(playerItemDataPairVector, true)

    local label = generatedLabelVector[1][1]

    if (label == "DoNotRecommend") then continue end

    table.insert(itemToShowArray, itemName)

    table.insert(itemDataMatrix, itemDataVector[1])

end

-- We need this to train our model even if the player does not perform the purchase. Every data counts!

hasPlayerPurchasedTheItemVector = TensorL:createTensor({#itemToShowDictionary, 1}) 

```

### Upon Item Purchase

```lua

local function onItemPurchase(itemName)

    local index = table.find(sortedItemToShowArray, itemName)

    if (not index) then return end

    hasPlayerPurchasedTheItemVector[index][1] = 1

end

```

### Upon Player Closing Shop GUI

```lua

local function onShopGUIClose()

    local costArray = RecommendationModel:train(itemDataMatrix, hasPlayerPurchasedTheItemVector)

end

```

This should give you a model that predicts a rough estimate on what they will likely to buy.

Then, you must save the model parameters to Roblox's DataStores for future use.

```lua

local ModelParameters = RecommendationModel:getModelParameters()

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

local costArray = RecommendationModel:setModelParameters(ModelParameters)

```

### Case 3: Every Player Uses The Same Global Model

Under this case, the procedure is the same to case 2 except that you need to:

* Load model parameters upon server start.

* Perform auto-save with the optional ability of merging with saved model parameters from other servers.

## Conclusion

This tutorial showed you on how to create item recommendation model that allows you to increase the likelihood the player will purchase an item. All you need is some data, some models and a bit of practice to get this right!

That's all for today and see you later!
