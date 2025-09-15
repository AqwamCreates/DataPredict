# Creating Reward-Maximization-Based Difficulty Generation Model

For this tutorial, we need multiple things to build our model, this includes:

* Neural Network Model

* Soft Actor-Critic Reinforcement Learning Model

* Diagonal Gaussian Policy Quick Setup

## Designing Our Feature Vector And Classes List

Before we start creating our model, we first need to visualize on how we will design our data and what actions the model could take to extend our players' play time.

### FeatureVector

```lua

-- We have five features with one "bias".

local playerDataVector = {
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

### defeatedEnemyDataVector

```lua

local defeatedEnemyDataVector = {

  {enemyMaximumHealth, enemyMaximumDamage, enemyCashAmount}

}


```

If you're concerned about that the model may produce wrong result heavily upon first start up, then you can use a randomized dataset to heavily skew the prediction. Then use this randomized dataset to pretrain the actor Neural Network before doing any real-time training and prediction. Below, we will show you how it is done.

```lua

local numberOfData = 100

local randomPlayerDataMatrix = TensorL:createRandomUniformTensor({numberOfData, 6}, -100, 100) -- 100 random data with 6 features (including one "bias")

local labelDataMatrix = TensorL:createTensor({numberOfData, 3}, 100)

```

However, this require setting the actor Neural Network's parameters to these settings temporarily so that it can be biased at start up as shown below.

```lua

ActorNeuralNetwork.maximumNumberOfIterations = 1000

ActorNeuralNetwork.learningRate = 0.3

```

## Constructing Our Model

Before we start training our model, we first need to build our model. We have split this to multiple subsections to make it easy to follow through.

### Constructing Our Neural Network

```lua 

local ActorNeuralNetwork = DataPredict.Model.NeuralNetwork.new({maximumNumberOfIterations = 1})

ActorNeuralNetwork:addLayer(5, true) -- Five features and one bias.

ActorNeuralNetwork:addLayer(3, false) --Three enemy features and no bias.

local CriticNeuralNetwork = DataPredict.Model.NeuralNetwork.new({maximumNumberOfIterations = 1})

CriticNeuralNetwork:addLayer(5, true) -- Five features and one bias.

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
    {
        enemyMaximumHealthStandardDeviation,
        enemyMaximumDamageStandardDeviation,
        enemyCashAmountStandardDeviation,
    }
}

-- Next, we'll insert actionStandardDeviationVector to our quick setup constructor.

local EnemyDataGenerationModel = DataPredict.QuickSetups.DiagonalGaussianPolicy.new({actionStandardDeviationVector = actionStandardDeviationVector})

-- Inserting our Deep Reinforcement Learning Model here.

EnemyDataGenerationModel:setModel(DeepReinforcementLearningModel)

```

## Training And Prediction

Because the way we have designed our Diagonal Gaussian Policy Quick Setup, you can immediately train while producing predictions for your player by calling reinforce() function.

This is because reinforce() function is responsible for producing prediction and perform pre-calculations at the same time as that is required to train our models.

```lua

-- Here, you notice that there is a reward value being inserted here. Generally, when you first call this, the reward value should be zero.

local generatedEnemyDataVector = EnemyDataGenerationModel:reinforce(playerDataVector, rewardValue)

```

## Rewarding Our Model

In order to assign the reward to that event is selected, we must first deploy the chosen event and observe if the player stayed for that action/event.

Below, it shows an example code for this.

```lua

local function run(Player)

    local playerDataVector = getPlayerDataVector(Player)

    local rewardValue = 0

    local generatedEnemyDataVector 

    while true do
    
        generatedEnemyDataVector  = EnemyDataGenerationModel:reinforce(playerDataVector, rewardValue)

        deployEventFunction = eventFunctionDictionary[eventName]

        if (deployEventFunction) then deployEventFunction() end

        task.wait(60)

        playerDataVector = getPlayerDataVector(Player)

        isPlayerInServer = checkIfPlayerIsInServer(Player)
        
        rewardValue = (isPlayerInServer and 10) or -50

    end

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

local DeepReinforcementLearningModel =  EnemyDataGenerationModel:getModel()

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

That's all for today!
