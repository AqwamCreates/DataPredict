local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

ActorCriticModel = {}

ActorCriticModel.__index = ActorCriticModel

local defaultNumberOfReinforcementsPerEpisode = 500

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

function ActorCriticModel.new(numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)
	
	local NewActorCriticModel = {}
	
	setmetatable(NewActorCriticModel, ActorCriticModel)
	
	NewActorCriticModel.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewActorCriticModel.epsilon = epsilon or defaultEpsilon

	NewActorCriticModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewActorCriticModel.discountFactor =  discountFactor or defaultDiscountFactor

	NewActorCriticModel.currentEpsilon = epsilon or defaultEpsilon

	NewActorCriticModel.previousFeatureVector = nil

	NewActorCriticModel.printReinforcementOutput = true

	NewActorCriticModel.currentNumberOfReinforcements = 0

	NewActorCriticModel.currentNumberOfEpisodes = 0
	
	return NewActorCriticModel
	
end

function ActorCriticModel:setActorModel(Model)
	
	self.ActorModel = Model
	
end

function ActorCriticModel:setCriticModel(Model)

	self.CriticModel = Model

end

function ActorCriticModel:setClassesList(classesList)
	
	self.ClassesList = classesList
	
end

function ActorCriticModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)



end

function ActorCriticModel:setExperienceReplay(ExperienceReplay)

	self.ExperienceReplay = ExperienceReplay

end

local function getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

function ActorCriticModel:setPrintReinforcementOutput(option)

	self.printReinforcementOutput = getBooleanOrDefaultOption(option, self.printReinforcementOutput)

end

function ActorCriticModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)
	
	if (self.ActorModel == nil) then error("No actor model!") end
	
	if (self.CriticModel == nil) then error("No critic model!") end

	if (self.currentNumberOfReinforcements >= self.numberOfReinforcementsPerEpisode) then

		self.currentNumberOfReinforcements = 0

		self.currentNumberOfEpisodes += 1

		self.currentEpsilon *= self.epsilonDecayFactor

	end

	self.currentNumberOfReinforcements += 1
	
	local action

	local actionVector

	local highestValue

	local highestValueVector

	local allOutputsMatrix

	local randomProbability = Random.new():NextNumber()

	if (randomProbability < self.currentEpsilon) then

		local randomNumber = Random.new():NextInteger(1, #self.ClassesList)

		action = self.ClassesList[randomNumber]

		allOutputsMatrix = AqwamMatrixLibrary:createMatrix(1, #self.ClassesList)

		allOutputsMatrix[1][randomNumber] = randomProbability

	else

		allOutputsMatrix = self:predict(currentFeatureVector, true)

		actionVector, highestValueVector = self:getLabelFromOutputMatrix(allOutputsMatrix)

		action = actionVector[1][1]

		highestValue = highestValueVector[1][1]

	end
	
	if (self.previousFeatureVector) then self:update(self.previousFeatureVector, action, rewardValue, currentFeatureVector) end

	if (self.ExperienceReplay) and (self.previousFeatureVector) then 

		self.ExperienceReplay:addExperience(self.previousFeatureVector, action, rewardValue, currentFeatureVector)

		self.ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

			self:update(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

		end)

	end

	self.previousFeatureVector = currentFeatureVector

	if (self.printReinforcementOutput) then print("Episode: " .. self.currentNumberOfEpisodes .. "\t\tEpsilon: " .. self.currentEpsilon .. "\t\tReinforcement Count: " .. self.currentNumberOfReinforcements) end

	if (returnOriginalOutput) then return allOutputsMatrix end

	return action, highestValue
	
end

function ActorCriticModel:getCurrentNumberOfEpisodes()

	return self.currentNumberOfEpisodes

end

function ActorCriticModel:getCurrentNumberOfReinforcements()

	return self.currentNumberOfReinforcements

end

function ActorCriticModel:reset()

	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil

	self.currentEpsilon = self.epsilon

	if (self.ExperienceReplay) then self.ExperienceReplay:reset() end

end

return ActorCriticModel
