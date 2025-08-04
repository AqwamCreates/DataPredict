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

local DeepReinforcementLearningBaseModel = require(script.Parent.DeepReinforcementLearningBaseModel)

DeepStateActionRewardStateActionModel = {}

DeepStateActionRewardStateActionModel.__index = DeepStateActionRewardStateActionModel

setmetatable(DeepStateActionRewardStateActionModel, DeepReinforcementLearningBaseModel)

local defaultLambda = 0

function DeepStateActionRewardStateActionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewDeepStateActionRewardStateActionModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepStateActionRewardStateActionModel, DeepStateActionRewardStateActionModel)
	
	NewDeepStateActionRewardStateActionModel:setName("DeepStateActionRewardStateAction")
	
	NewDeepStateActionRewardStateActionModel.lambda = parameterDictionary.lambda or defaultLambda
	
	NewDeepStateActionRewardStateActionModel.eligibilityTraceMatrix = parameterDictionary.eligibilityTraceMatrix

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

			local eligibilityTraceMatrix = NewDeepStateActionRewardStateActionModel.eligibilityTraceMatrix

			if (not eligibilityTraceMatrix) then eligibilityTraceMatrix = AqwamTensorLibrary:createTensor({1, #ClassesList}, 0) end

			eligibilityTraceMatrix = AqwamTensorLibrary:multiply(eligibilityTraceMatrix, discountFactor * lambda)

			eligibilityTraceMatrix[1][actionIndex] = eligibilityTraceMatrix[1][actionIndex] + 1

			temporalDifferenceErrorVector = AqwamTensorLibrary:multiply(temporalDifferenceErrorVector, eligibilityTraceMatrix)

			NewDeepStateActionRewardStateActionModel.eligibilityTraceMatrix = eligibilityTraceMatrix

		end
		
		local negatedTemporalDifferenceErrorVector = AqwamTensorLibrary:unaryMinus(temporalDifferenceErrorVector) -- The original non-deep SARSA version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the error vector to make the neural network to perform gradient ascent.
		
		Model:forwardPropagate(previousFeatureVector, true)

		Model:update(negatedTemporalDifferenceErrorVector, true)
		
		return temporalDifferenceErrorVector

	end)
	
	NewDeepStateActionRewardStateActionModel:setEpisodeUpdateFunction(function(terminalStateValue) 
		
		NewDeepStateActionRewardStateActionModel.eligibilityTraceMatrix = nil
		
	end)
	
	NewDeepStateActionRewardStateActionModel:setResetFunction(function() 
		
		NewDeepStateActionRewardStateActionModel.eligibilityTraceMatrix = nil
		
	end)

	return NewDeepStateActionRewardStateActionModel

end

function DeepStateActionRewardStateActionModel:setParameters(discountFactor)

	self.discountFactor = discountFactor or self.discountFactor

end

return DeepStateActionRewardStateActionModel
