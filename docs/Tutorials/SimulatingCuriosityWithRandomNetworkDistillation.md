# Simulating Curiosity With Random Network Distillation

Majority of the time, rewards are received from the environment when performing certain actions. These are usually handcrafted by programmers. We call them external rewards.

However, what if there are no external rewards and we want to encourage our AIs to explore a map?

Then, we need to find a way for our AIs to be "curious".

# The Random Network Distillation

Random Network Distillation uses neural networks to generate internal rewards to stimulate curiousity. At the start, it will encourage exploration. But over time, it sees more than enough similar data and this will cause the model to stop exploring.

Under here, this is how you integrate your reinforcement learning algorithms with the random network distillation.

```lua

-- Initializing our NeuralNetwork.

local NeuralNetwork = DataPredict.Models.NeuralNetwork.new()

NeuralNetwork:addLayer(10, true, "None")

NeuralNetwork:addLayer(4, true, "LeakyReLU")

-- Initializing our RandomNetworkDistillation.

local RandomNetworkDistillation = DataPredict.ReinforcementLearningStrategies.RandomNetworkDistillation.new()

RandomNetworkDistillation:setModel(NeuralNetwork)

-- Initializing our ReinforcementLearningQuickSetup.

local QLearningNeuralNetworkQuickSetup = DataPredict.QuickSetups.CategoricalPolicy.new()

QLearningNeuralNetworkQuickSetup:setModel(QLearningNeuralNetwork)

QLearningNeuralNetworkQuickSetup:setClassesList({1, 2, 3, 4})

-- Creating a simple function when receiving environment vector received.

local function onEnvironmentFeatureVectorReceived(environmentFeatureVector)

  local rewardVector = RandomNetworkDistillation:generate(environmentFeatureVector)

  local internalReward = rewardVector[1][1]

  local action = QLearningNeuralNetworkQuickSetup:reinforce(environmentFeatureVector, internalReward)

  return action

end

```

As you can see, creating a random network distillation object is pretty similar to creating neural networks. The difference lies on the number of output it produces, where the random network distillation always produce one output. The number of neurons you add at the final layer does not matter, as generate() function will convert multiple outputs to one.

# Discouraging Exploration

You can discourage the AI from exploring by making the internal reward value to negative. An example is shown below:

```lua

local rewardVector = RandomNetworkDistillation:generate(environmentFeatureVector)

local internalReward = rewardVector[1][1]

local negativeInternalReward = -internalReward

```

This is particularly useful if you want to prevent AI from doing things like:

* Keep walking forward even if it is blocked by a wall.

* Moves to areas that actively harms the AI.

* And many others!

Anyways, that's all for today!
