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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local ReinforcementLearningBaseModel = require(script.Parent.ReinforcementLearningBaseModel)

DeepDoubleStateActionRewardStateActionModel = {}

DeepDoubleStateActionRewardStateActionModel.__index = DeepDoubleStateActionRewardStateActionModel

setmetatable(DeepDoubleStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local defaultAveragingRate = 0.995

local defaultLambda = 0

local function rateAverageModelParameters(averagingRate, TargetModelParameters, PrimaryModelParameters)

	local averagingRateComplement = 1 - averagingRate

	for layer = 1, #TargetModelParameters, 1 do

		local TargetModelParametersPart = AqwamTensorLibrary:multiply(averagingRate, TargetModelParameters[layer])

		local PrimaryModelParametersPart = AqwamTensorLibrary:multiply(averagingRateComplement, PrimaryModelParameters[layer])

		TargetModelParameters[layer] = AqwamTensorLibrary:add(TargetModelParametersPart, PrimaryModelParametersPart)

	end

	return TargetModelParameters

end

function DeepDoubleStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepDoubleStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepDoubleStateActionRewardStateActionModel, DeepDoubleStateActionRewardStateActionModel)
	
	NewDeepDoubleStateActionRewardStateActionModel:setName("DeepDoubleStateActionRewardStateActionV2")

	NewDeepDoubleStateActionRewardStateActionModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate
	
	NewDeepDoubleStateActionRewardStateActionModel.lambda = parameterDictionary.lambda or defaultLambda

	NewDeepDoubleStateActionRewardStateActionModel.eligibilityTrace = parameterDictionary.eligibilityTrace

	NewDeepDoubleStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local Model = NewDeepDoubleStateActionRewardStateActionModel.Model
		
		local discountFactor = NewDeepDoubleStateActionRewardStateActionModel.discountFactor

		local lambda = NewDeepDoubleStateActionRewardStateActionModel.lambda
		
		local PrimaryModelParameters = Model:getModelParameters(true)

		if (PrimaryModelParameters) then 
			
			Model:generateLayers()
			
			PrimaryModelParameters = Model:getModelParameters(true)
			
		end
		
		local qVector = Model:forwardPropagate(currentFeatureVector, true)

		local discountedQVector = AqwamTensorLibrary:multiply(discountFactor, qVector, (1 - terminalStateValue))

		local targetVector = AqwamTensorLibrary:add(rewardValue, discountedQVector)

		local previousQVector = Model:forwardPropagate(previousFeatureVector)

		local temporalDifferenceErrorVector = AqwamTensorLibrary:subtract(targetVector, previousQVector)
		
		if (lambda ~= 0) then

			local ClassesList = Model:getClassesList()

			local actionIndex = table.find(ClassesList, action)

			local eligibilityTrace = NewDeepDoubleStateActionRewardStateActionModel.eligibilityTrace

			if (not eligibilityTrace) then eligibilityTrace = AqwamTensorLibrary:createTensor({1, #ClassesList}, 0) end

			eligibilityTrace = AqwamTensorLibrary:multiply(eligibilityTrace, discountFactor * lambda)

			eligibilityTrace[1][actionIndex] = eligibilityTrace[1][actionIndex] + 1

			temporalDifferenceErrorVector = AqwamTensorLibrary:multiply(temporalDifferenceErrorVector, eligibilityTrace)

			NewDeepDoubleStateActionRewardStateActionModel.eligibilityTrace = eligibilityTrace

		end

		Model:forwardPropagate(previousFeatureVector, true, true)

		Model:backwardPropagate(temporalDifferenceErrorVector, true)
		
		local TargetModelParameters = Model:getModelParameters(true)

		TargetModelParameters = rateAverageModelParameters(NewDeepDoubleStateActionRewardStateActionModel.averagingRate, TargetModelParameters, PrimaryModelParameters)

		Model:setModelParameters(TargetModelParameters, true)
		
		return temporalDifferenceErrorVector

	end)
	
	NewDeepDoubleStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		NewDeepDoubleStateActionRewardStateActionModel.eligibilityTrace = nil
		
	end)

	NewDeepDoubleStateActionRewardStateActionModel:setResetFunction(function()
		
		NewDeepDoubleStateActionRewardStateActionModel.eligibilityTrace = nil
		
	end)

	return NewDeepDoubleStateActionRewardStateActionModel

end

return DeepDoubleStateActionRewardStateActionModel