# Creating Simple Play Time Maximization Model

For this tutorial, we need multiple things to build our model, this includes:

* Tabular Reinforcement Learning Model (Tabular Q-Learning or Tabular SARSA)

* Categorical Policy Quick Setup

## Designing Our Feature Vector And Classes List

Before we start creating our model, we first need to visualize on how we will design our data and what actions the model could take to extend our players' play time.

### StatesList

```lua

-- We have five features with one "bias".

local StatesList = {

    "PlayerAwayFromKeyboard",
    "PlayerIdle".
    "PlayerRewarded",
    "PlayerPickingUpItem",
    "PlayerActiveForQuest"
    "PlayerActiveAgainstEnemy"
    "PlayerActiveAgainstEnemyBoss"
    "PlayerLeft"
    "PlayerLostConnection"

}

```

### ActionsList

```lua

local ActionsList = {

  "NoEvent",
  "FreeGiftEvent",
  "ResourceMultiplierEvent",
  "QuestEvent",
  "ItemSpawnEvent",
  "BossSpawnEvent",
  "LimitedTimeQuestEvent",
  "LimitedTimeItemSpawnEvent",
  "LimitedTimeBossSpawnEvent",

}

```

## Constructing Our Model

### Constructing Our Tabular Reinforcement Learning Model

```lua


--[[

    You can use Tabular SARSA here for safer learning.

    However, because our model isn't that complex, it is better to use the model that choses
    the best actions like Tabular Q-learning.

    We can then further improve it by using eligibility traces to keep track on what actions is to
    be "blamed" for causing the player to reach next state.

    Feeling bold as well? Let's add optimizers to the mix reserved for speeding up neural network learning, but I over engineered
    the tabular reinforcement learning models so that these can use optimizers. 

--]]

local EligibilityTrace = DataPredict.EligibilityTrace.AccumulatingTrace.new()

--[[

    Got plenty of optimizers here, but AdaptiveMomentEstimation (Adam) is always
    the best performing on in deep reinforcement learning research.

--]]

local Optimizer = DataPredict.

local TabularReinforcementLearningModel = DataPredict.Model.TabularQLearning.new({StatesList = StatesList, ActionsList = ActionsList, EligibilityTrace = EligibilityTrace})

```

### Constructing Our Categorical Policy Quick Setup Model

This part makes it easier for us to set up our model, but it is not strictly necessary. However, I do recommend you to use them as they contain built-in functions for handing training and predictions.

```lua

local PlayTimeMaximizationModel = DataPredict.QuickSetups.CategoricalPolicy.new()

-- Inserting our Deep Reinforcement Learning Model here.

PlayTimeMaximizationModel:setModel(TabularReinforcementLearningModel)

```

## Training And Prediction

Because the way we have designed our Categorical Policy Quick Setup, you can immediately train while producing predictions for your player by calling reinforce() function.

This is because reinforce() function is responsible for producing prediction and perform pre-calculations at the same time as that is required to train our models.

```lua

-- Here, you notice that there is a reward value being inserted here. Generally, when you first call this, the reward value should be zero.

local eventName = PlayTimeMaximizationModel:reinforce(playerDataVector, rewardValue)

```

## Rewarding Our Model

In order to assign the reward to that event is selected, we must first deploy the chosen event and observe if the player stayed for that event.

Below, it shows an example code for this.

```lua

local eventFunctionDictionary = {

  ["NoEvent"] = nil,
  ["ResourceMultiplierEvent"] = resourceMultiplierEvent,
  ["QuestEvent"] = questEvent,
  ["ItemSpawnEvent"] = itemSpawnEvent,
  ["BossSpawnEvent"] = bossSpawnEvent,
  ["LimitedTimeQuestEvent"] = limitedTimeQuestEvent,
  ["LimitedTimeItemSpawnEvent"] = limitedTimeItemSpawnEvent,
  ["LimitedTimeBossSpawnEvent"] = limitedTimeBossSpawnEvent,

}

local function run(Player)

    local isPlayerInServer = true

    local rewardValue = 0

    local playerState

    local eventName

    local eventFunction

    while isPlayerInServer do

        playerState = getPlayerState(Player)
    
        eventName = PlayTimeMaximizationModel:reinforce(playerState, rewardValue)

        eventFunction = eventFunctionDictionary[eventName]

        if (eventFunction) then eventFunction() end

        task.wait(30)

        isPlayerInServer = checkIfPlayerIsInServer(Player)

        -- Player leaving the game is more of a "rarer" and "extremely undesirable" event, therefore a very large negative value is used.

        rewardValue = (isPlayerInServer and 20) or -100

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

ModelParameters = PlayTimeMaximizationModel:getModelParameters()

PlayTimeMaximizationModel:setModelParameters(ModelParameters)

```

### Case 3: Every Player Uses The Same Global Model

Under this case, the procedure is the same to case 2 except that you need to:

* Load model parameters upon server start.

* Perform auto-save with the optional ability of merging with saved model parameters from other servers.

That's all for today!
