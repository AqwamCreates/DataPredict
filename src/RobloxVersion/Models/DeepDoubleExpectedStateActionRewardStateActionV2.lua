--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

DeepDoubleExpectedStateActionRewardStateActionModel = {}

DeepDoubleExpectedStateActionRewardStateActionModel.__index = DeepDoubleExpectedStateActionRewardStateActionModel

setmetatable(DeepDoubleExpectedStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local defaultEpsilon = 0.5

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

function DeepDoubleExpectedStateActionRewardStateActionModel.new(maxNumberOfIterations, epsilon, averagingRate, discountFactor)

	local NewDeepDoubleExpectedStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(maxNumberOfIterations, discountFactor)

	setmetatable(NewDeepDoubleExpectedStateActionRewardStateActionModel, DeepDoubleExpectedStateActionRewardStateActionModel)
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel.epsilon = epsilon or defaultEpsilon
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel.averagingRate = averagingRate or defaultAveragingRate

	NewDeepDoubleExpectedStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local Model = NewDeepDoubleExpectedStateActionRewardStateActionModel.Model
		
		local PrimaryModelParameters = Model:getModelParameters(true)

		if (not PrimaryModelParameters) then 
			
			Model:generateLayers()
			
			PrimaryModelParameters = Model:getModelParameters(true)
			
		end

		local expectedQValue = 0

		local numberOfGreedyActions = 0
		
		local ClassesList = Model:getClassesList()

		local numberOfActions = #ClassesList

		local actionIndex = table.find(ClassesList, action)

		local previousVector = Model:forwardPropagate(previousFeatureVector)

		local targetVector = Model:forwardPropagate(currentFeatureVector)
		
		local maxQValue = targetVector[1][actionIndex]

		for i = 1, numberOfActions, 1 do

			if (targetVector[1][i] ~= maxQValue) then
				
				numberOfGreedyActions = numberOfGreedyActions + 1
				
			end

		end

		local nonGreedyActionProbability = NewDeepDoubleExpectedStateActionRewardStateActionModel.epsilon / numberOfActions

		local greedyActionProbability = ((1 - NewDeepDoubleExpectedStateActionRewardStateActionModel.epsilon) / numberOfGreedyActions) + nonGreedyActionProbability

		for _, qValue in ipairs(targetVector[1]) do

			if (qValue == maxQValue) then

				expectedQValue = expectedQValue + (qValue * greedyActionProbability)

			else

				expectedQValue = expectedQValue + (qValue * nonGreedyActionProbability)

			end

		end

		local targetValue = rewardValue + (NewDeepDoubleExpectedStateActionRewardStateActionModel.discountFactor * expectedQValue)

		local lastValue = previousVector[1][actionIndex]

		local temporalDifferenceError = targetValue - lastValue

		local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfActions, 0)

		lossVector[1][actionIndex] = temporalDifferenceError
		
		Model:forwardPropagate(previousFeatureVector, true, true)

		Model:backwardPropagate(lossVector, true)

		local TargetModelParameters = Model:getModelParameters(true)

		TargetModelParameters = rateAverageModelParameters(NewDeepDoubleExpectedStateActionRewardStateActionModel.averagingRate, TargetModelParameters, PrimaryModelParameters)

		Model:setModelParameters(TargetModelParameters, true)
		
		return temporalDifferenceError

	end)
	
	NewDeepDoubleExpectedStateActionRewardStateActionModel:setEpisodeUpdateFunction(function() end)

	NewDeepDoubleExpectedStateActionRewardStateActionModel:setResetFunction(function() end)

	return NewDeepDoubleExpectedStateActionRewardStateActionModel

end

function DeepDoubleExpectedStateActionRewardStateActionModel:setParameters(epsilon, averagingRate, discountFactor)
	
	self.epsilon = epsilon or self.epsilon

	self.discountFactor = discountFactor or self.discountFactor

	self.averagingRate = averagingRate or self.averagingRate

end

return DeepDoubleExpectedStateActionRewardStateActionModel