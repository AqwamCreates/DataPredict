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

DeepStateActionRewardStateActionModel = {}

DeepStateActionRewardStateActionModel.__index = DeepStateActionRewardStateActionModel

setmetatable(DeepStateActionRewardStateActionModel, ReinforcementLearningBaseModel)

local defaultLambda = 0

function DeepStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepStateActionRewardStateActionModel = ReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepStateActionRewardStateActionModel, DeepStateActionRewardStateActionModel)
	
	NewDeepStateActionRewardStateActionModel:setName("DeepStateActionRewardStateAction")
	
	NewDeepStateActionRewardStateActionModel.lambda = parameterDictionary.lambda or defaultLambda
	
	NewDeepStateActionRewardStateActionModel.eligibilityTrace = parameterDictionary.eligibilityTrace

	NewDeepStateActionRewardStateActionModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)
		
		local Model = NewDeepStateActionRewardStateActionModel.Model
		
		local discountFactor = NewDeepStateActionRewardStateActionModel.discountFactor
		
		local lambda = NewDeepStateActionRewardStateActionModel.lambda

		local qVector = Model:forwardPropagate(currentFeatureVector)

		local discountedQVector = AqwamTensorLibrary:multiply(discountFactor, qVector, (1 - terminalStateValue))

		local targetQVector = AqwamTensorLibrary:add(rewardValue, discountedQVector)

		local previousQVector = Model:forwardPropagate(previousFeatureVector)

		local temporalDifferenceErrorVector = AqwamTensorLibrary:subtract(targetQVector, previousQVector)
		
		if (lambda ~= 0) then
			
			local ClassesList = Model:getClassesList()
			
			local actionIndex = table.find(ClassesList, action)

			local eligibilityTrace = NewDeepStateActionRewardStateActionModel.eligibilityTrace

			if (not eligibilityTrace) then eligibilityTrace = AqwamTensorLibrary:createTensor({1, #ClassesList}, 0) end

			eligibilityTrace = AqwamTensorLibrary:multiply(eligibilityTrace, discountFactor * lambda)

			eligibilityTrace[1][actionIndex] = eligibilityTrace[1][actionIndex] + 1

			temporalDifferenceErrorVector = AqwamTensorLibrary:multiply(temporalDifferenceErrorVector, eligibilityTrace)

			NewDeepStateActionRewardStateActionModel.eligibilityTrace = eligibilityTrace

		end
		
		Model:forwardPropagate(previousFeatureVector, true, true)

		Model:backwardPropagate(temporalDifferenceErrorVector, true)
		
		return temporalDifferenceErrorVector

	end)
	
	NewDeepStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		NewDeepStateActionRewardStateActionModel.eligibilityTrace = nil
		
	end)
	
	NewDeepStateActionRewardStateActionModel:setResetFunction(function() 
		
		NewDeepStateActionRewardStateActionModel.eligibilityTrace = nil
		
	end)

	return NewDeepStateActionRewardStateActionModel

end

function DeepStateActionRewardStateActionModel:setParameters(discountFactor)

	self.discountFactor = discountFactor or self.discountFactor

end

return DeepStateActionRewardStateActionModel