--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

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

local DeepReinforcementLearningBaseModel = require(script.Parent.DeepReinforcementLearningBaseModel)

DeepDoubleStateActionRewardStateActionModel = {}

DeepDoubleStateActionRewardStateActionModel.__index = DeepDoubleStateActionRewardStateActionModel

setmetatable(DeepDoubleStateActionRewardStateActionModel, DeepReinforcementLearningBaseModel)

local defaultAveragingRate = 0.995

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

	local NewDeepDoubleStateActionRewardStateActionModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepDoubleStateActionRewardStateActionModel, DeepDoubleStateActionRewardStateActionModel)
	
	NewDeepDoubleStateActionRewardStateActionModel:setName("DeepDoubleStateActionRewardStateActionV2")

	NewDeepDoubleStateActionRewardStateActionModel.averagingRate = parameterDictionary.averagingRate or defaultAveragingRate

	NewDeepDoubleStateActionRewardStateActionModel.EligibilityTrace = parameterDictionary.EligibilityTrace

	NewDeepDoubleStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local Model = NewDeepDoubleStateActionRewardStateActionModel.Model
		
		local discountFactor = NewDeepDoubleStateActionRewardStateActionModel.discountFactor

		local EligibilityTrace = NewDeepDoubleStateActionRewardStateActionModel.EligibilityTrace
		
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
		
		if (EligibilityTrace) then

			local ClassesList = Model:getClassesList()

			local actionIndex = table.find(ClassesList, action)

			EligibilityTrace:increment(actionIndex, discountFactor, {1, #ClassesList})

			temporalDifferenceErrorVector = EligibilityTrace:calculate(temporalDifferenceErrorVector)

		end
		
		local negatedTemporalDifferenceErrorVector = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorVector) -- The original non-deep SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error vector to make the neural network to perform gradient ascent.

		Model:forwardPropagate(previousFeatureVector, true)

		Model:update(negatedTemporalDifferenceErrorVector, true)
		
		local TargetModelParameters = Model:getModelParameters(true)

		TargetModelParameters = rateAverageModelParameters(NewDeepDoubleStateActionRewardStateActionModel.averagingRate, TargetModelParameters, PrimaryModelParameters)

		Model:setModelParameters(TargetModelParameters, true)
		
		return temporalDifferenceErrorVector

	end)
	
	NewDeepDoubleStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		NewDeepDoubleStateActionRewardStateActionModel.EligibilityTrace:reset()
		
	end)

	NewDeepDoubleStateActionRewardStateActionModel:setResetFunction(function()
		
		NewDeepDoubleStateActionRewardStateActionModel.EligibilityTrace:reset()
		
	end)

	return NewDeepDoubleStateActionRewardStateActionModel

end

return DeepDoubleStateActionRewardStateActionModel
