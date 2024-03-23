local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

ReinforcementLearningActorCriticNeuralNetworkBaseModel = {}

ReinforcementLearningActorCriticNeuralNetworkBaseModel.__index = ReinforcementLearningActorCriticNeuralNetworkBaseModel

local defaultNumberOfReinforcementsPerEpisode = 500

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

function ReinforcementLearningActorCriticNeuralNetworkBaseModel.new(discountFactor)
	
	local NewReinforcementLearningActorCriticNeuralNetworkBaseModel = {}
	
	setmetatable(NewReinforcementLearningActorCriticNeuralNetworkBaseModel, ReinforcementLearningActorCriticNeuralNetworkBaseModel)
	
	NewReinforcementLearningActorCriticNeuralNetworkBaseModel.discountFactor = discountFactor or defaultDiscountFactor

	NewReinforcementLearningActorCriticNeuralNetworkBaseModel.printReinforcementOutput = true
	
	return NewReinforcementLearningActorCriticNeuralNetworkBaseModel
	
end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:setParameters(discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor
	
end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:setActorModel(ActorModel)
	
	ActorModel:setPrintOutput(false)
	
	self.ActorModel = ActorModel
	
end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:setCriticModel(CriticModel)
	
	CriticModel:setPrintOutput(false)

	self.CriticModel = CriticModel
	
end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:setUpdateFunction(updateFunction)

	self.updateFunction = updateFunction

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:predict(featureVector, returnOriginalOutput)
	
	return self.ActorModel:predict(featureVector, returnOriginalOutput)
	
end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)

	self.updateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector)

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:episodeUpdate()

	local episodeUpdateFunction = self.episodeUpdateFunction

	if not episodeUpdateFunction then return end

	episodeUpdateFunction()

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:getActorModel()
	
	return self.ActorModel
	
end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:getCriticModel()

	return self.CriticModel

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:extendResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:reset()
	
	local ActorModel = self.ActorModel
	
	local CriticModel = self.CriticModel
	
	if (ActorModel) then ActorModel:reset() end
	
	if (CriticModel) then CriticModel:reset() end

	if (self.resetFunction) then self.resetFunction() end

end

function ReinforcementLearningActorCriticNeuralNetworkBaseModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return ReinforcementLearningActorCriticNeuralNetworkBaseModel
