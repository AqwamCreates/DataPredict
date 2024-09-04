--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local ReinforcementLearningBaseModel = require("Model_ReinforcementLearningBaseModel")

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

	NewDeepDoubleQLearningModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewDeepDoubleQLearningModel.Model
		
		local PrimaryModelParameters = Model:getModelParameters(true)

		if (PrimaryModelParameters == nil) then 
			
			Model:generateLayers()
			PrimaryModelParameters = Model:getModelParameters(true)
			
		end

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

		Model:backwardPropagate(lossVector, true)

		local TargetModelParameters = Model:getModelParameters(true)

		TargetModelParameters = rateAverageModelParameters(NewDeepDoubleQLearningModel.averagingRate, TargetModelParameters, PrimaryModelParameters)

		Model:setModelParameters(TargetModelParameters, true)
		
		return temporalDifferenceError

	end)
	
	NewDeepDoubleQLearningModel:setCategoricalEpisodeUpdateFunction(function() end)

	NewDeepDoubleQLearningModel:setCategoricalResetFunction(function() end)
	
	return NewDeepDoubleQLearningModel

end

function DeepDoubleQLearningModel:setParameters(averagingRate, discountFactor)

	self.discountFactor =  discountFactor or self.discountFactor
	
	self.averagingRate = averagingRate or self.averagingRate

end

return DeepDoubleQLearningModel
