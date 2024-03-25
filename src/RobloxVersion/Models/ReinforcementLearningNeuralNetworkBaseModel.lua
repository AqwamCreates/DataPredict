local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local NeuralNetworkModel = require(script.Parent.NeuralNetwork)

ReinforcementLearningNeuralNetworkBaseModel = {}

ReinforcementLearningNeuralNetworkBaseModel.__index = ReinforcementLearningNeuralNetworkBaseModel

setmetatable(ReinforcementLearningNeuralNetworkBaseModel, NeuralNetworkModel)

local defaultDiscountFactor = 0.95

local defaultMaxNumberOfIterations = 1

function ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, discountFactor)
	
	maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	local NewReinforcementLearningNeuralNetworkBaseModel = NeuralNetworkModel.new(maxNumberOfIterations)

	NewReinforcementLearningNeuralNetworkBaseModel:setPrintOutput(false)

	setmetatable(NewReinforcementLearningNeuralNetworkBaseModel, ReinforcementLearningNeuralNetworkBaseModel)

	NewReinforcementLearningNeuralNetworkBaseModel.discountFactor =  discountFactor or defaultDiscountFactor

	return NewReinforcementLearningNeuralNetworkBaseModel
	
end

function ReinforcementLearningNeuralNetworkBaseModel:setParameters(maxNumberOfIterations, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.discountFactor =  discountFactor or self.discountFactor

end

function ReinforcementLearningNeuralNetworkBaseModel:setUpdateFunction(updateFunction)
	
	self.updateFunction = updateFunction
	
end

function ReinforcementLearningNeuralNetworkBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)
	
	self.episodeUpdateFunction = episodeUpdateFunction
	
end

function ReinforcementLearningNeuralNetworkBaseModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
	return self.updateFunction(previousFeatureVector, action, rewardValue, currentFeatureVector)
	
end

function ReinforcementLearningNeuralNetworkBaseModel:episodeUpdate()
	
	local episodeUpdateFunction = self.episodeUpdateFunction
	
	if not episodeUpdateFunction then return end
	
	episodeUpdateFunction()
	
end

function ReinforcementLearningNeuralNetworkBaseModel:extendResetFunction(resetFunction)
	
	self.resetFunction = resetFunction
	
end

function ReinforcementLearningNeuralNetworkBaseModel:reset()

	for i, Optimizer in ipairs(self.OptimizerTable) do

		if Optimizer then Optimizer:reset() end

	end
	
	if (self.resetFunction) then self.resetFunction() end

end

return ReinforcementLearningNeuralNetworkBaseModel
