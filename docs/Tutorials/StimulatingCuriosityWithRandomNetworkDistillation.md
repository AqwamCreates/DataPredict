# Stimulating Curiosity With Random Network Distillation

Majority of the time, rewards are received from the environment when performing certain actions. These are usually handcrafted by programmers. We call them external rewards.

However, what if there are no external rewards and we want to encourage our AIs to explore a map?

Then, we need to find a way for our AIs to be "curious".

# The Random Network Distillation

Random Network Distillation uses neural networks to generate internal rewards to stimulate curiousity. At the start, it will encourage exploration. But over time, it sees more than enough similar data and this will discourage exploration.

Under here, this is how you integrate your reinforcement learning algorithms with the random network distillation.

```lua

-- Initializing our RandomNetworkDistillation

local RandomNetworkDistillation = DataPredict.Others.RandomNetworkDistillation.new()

RandomNetworkDistillation:addLayer(10, true, "LeakyReLU")

RandomNetworkDistillation:addLayer(4, true, "Sigmoid")

-- Initializing our QLearningNeuralNetwork

local QLearningNeuralNetwork = DataPredict.Model.QLearning.new()

QLearningNeuralNetwork:addLayer(10, true, "LeakyReLU")

QLearningNeuralNetwork:addLayer(4, true, "StableSoftmax")

-- Creating A simple function when receiving environment vector received.

local function onEnvironmentVectorReceived(environmentVector)

  local internalReward = RandomNetworkDistillation:generateReward(environmentVector)

  local action = QLearningNeuralNetwork:reinforce(environmentVector, internalReward)

  return action

end

```

As you can see, creating a random network distillation object is pretty similar to creating neural networks. The difference lies on the number of output it produces, where the random network distillation always produce one output. The number of neurons you add at the final layer does not matter, as generateReward() function will convert multiple outputs to one.

# Discouraging Exploration

You can discourage the AI from exploring by making the internal reward value to negative. An example is shown below:

```lua

 local internalReward = -RandomNetworkDistillation:generate(environmentVector)

```

This is particularly useful if you want to prevent AI from doing things like:

* Keep walking forward even if it is blocked by a wall.

* Moves to area that actively harms the AI.

* And many others!

Anyways, that's all for today!
