local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

DeepDoubleQLearningModel = {}

DeepDoubleQLearningModel.__index = DeepDoubleQLearningModel

setmetatable(DeepDoubleQLearningModel, ReinforcementLearningBaseModel)

local defaultAveragingRate = 0.01

local function rateAverageModelParameters(averagingRate, TargetModelParameters, PrimaryModelParameters)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetModelParameters, 1 do

		local TargetModelParametersPart = AqwamMatrixLibrary:multiply(averagingRate, TargetModelParameters[layer])

		local PrimaryModelParametersPart = AqwamMatrixLibrary:multiply(averagingRateComplement, PrimaryModelParameters[layer])

		TargetModelParameters[layer] = AqwamMatrixLibrary:add(TargetModelParametersPart, PrimaryModelParametersPart)

	end

	return TargetModelParameters

end

function DeepDoubleQLearningModel.new(averagingRate, discountFactor)

	local NewDeepDoubleQLearningModel = ReinforcementLearningBaseModel.new(discountFactor)

	setmetatable(NewDeepDoubleQLearningModel, DeepDoubleQLearningModel)
	
	NewDeepDoubleQLearningModel.averagingRate = averagingRate or defaultAveragingRate

	NewDeepDoubleQLearningModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewDeepDoubleQLearningModel.Model

		if (NewDeepDoubleQLearningModel.Model == nil) then Model:generateLayers() end

		local PrimaryModelParameters = Model:getModelParameters(true)

		local predictedValue, maxQValue = Model:predict(currentFeatureVector)

		local targetValue = rewardValue + (NewDeepDoubleQLearningModel.discountFactor * maxQValue[1][1])

		local previousVector = Model:predict(previousFeatureVector, true)
		
		local ClassesList = Model:getClassesList()

		local actionIndex = table.find(ClassesList, action)

		local lastValue = previousVector[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue
		
		local numberOfClasses = #ClassesList

		local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfClasses)

		lossVector[1][actionIndex] = temporalDifferenceError

		Model:forwardPropagate(previousFeatureVector, true)

		Model:backPropagate(lossVector, true)

		local TargetModelParameters = Model:getModelParameters(true)

		TargetModelParameters = rateAverageModelParameters(NewDeepDoubleQLearningModel.averagingRate, TargetModelParameters, PrimaryModelParameters)

		Model:setModelParameters(TargetModelParameters, true)
		
		return temporalDifferenceError

	end)
	
	return NewDeepDoubleQLearningModel

end

function DeepDoubleQLearningModel:setParameters(averagingRate, discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor
	
	self.averagingRate = averagingRate or self.averagingRate

end

return DeepDoubleQLearningModel
