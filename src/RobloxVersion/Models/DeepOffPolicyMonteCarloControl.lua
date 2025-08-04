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

DeepOffPolicyMonteCarloControlModel = {}

DeepOffPolicyMonteCarloControlModel.__index = DeepOffPolicyMonteCarloControlModel

setmetatable(DeepOffPolicyMonteCarloControlModel, DeepReinforcementLearningBaseModel)

local defaultTargetPolicyFunction = "StableSoftmax"

local targetPolicyFunctionList = {

	["Greedy"] = function (actionVector)

		local targetActionVector = AqwamTensorLibrary:createTensor({1, #actionVector[1]}, 0)

		local highestActionValue = -math.huge

		local indexWithHighestActionValue

		for i, actionValue in ipairs(actionVector[1]) do

			if (actionValue > highestActionValue) then

				highestActionValue = actionValue

				indexWithHighestActionValue = i

			end

		end

		targetActionVector[1][indexWithHighestActionValue] = highestActionValue

		return targetActionVector

	end,

	["Softmax"] = function (actionVector) -- Apparently Lua doesn't really handle very small values such as math.exp(-1000), so I added a more stable computation exp(a) / exp(b) -> exp (a - b).

		local exponentActionVector = AqwamTensorLibrary:applyFunction(math.exp, actionVector)

		local exponentActionSumVector = AqwamTensorLibrary:sum(exponentActionVector, 2)

		local targetActionVector = AqwamTensorLibrary:divide(exponentActionVector, exponentActionSumVector)

		return targetActionVector

	end,

	["StableSoftmax"] = function (actionVector)

		local highestActionValue = AqwamTensorLibrary:findMaximumValue(actionVector)

		local subtractedZVector = AqwamTensorLibrary:subtract(actionVector, highestActionValue)

		local exponentActionVector = AqwamTensorLibrary:applyFunction(math.exp, subtractedZVector)

		local exponentActionSumVector = AqwamTensorLibrary:sum(exponentActionVector, 2)

		local targetActionVector = AqwamTensorLibrary:divide(exponentActionVector, exponentActionSumVector)

		return targetActionVector

	end,

}

function DeepOffPolicyMonteCarloControlModel.new(parameterDictionary)

	local NewDeepOffPolicyMonteCarloControlModel = DeepReinforcementLearningBaseModel.new(parameterDictionary)

	setmetatable(NewDeepOffPolicyMonteCarloControlModel, DeepOffPolicyMonteCarloControlModel)

	NewDeepOffPolicyMonteCarloControlModel:setName("DeepOffPolicyMonteCarloControl")

	NewDeepOffPolicyMonteCarloControlModel.targetPolicyFunction = parameterDictionary.targetPolicyFunction or defaultTargetPolicyFunction

	local featureVectorHistory = {}

	local actionVectorHistory = {}

	local rewardValueHistory = {}

	NewDeepOffPolicyMonteCarloControlModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector, terminalStateValue)

		local actionVector = NewDeepOffPolicyMonteCarloControlModel.Model:forwardPropagate(previousFeatureVector)

		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(actionVectorHistory, actionVector)

		table.insert(rewardValueHistory, rewardValue)

	end)

	NewDeepOffPolicyMonteCarloControlModel:setEpisodeUpdateFunction(function(terminalStateValue)

		local Model = NewDeepOffPolicyMonteCarloControlModel.Model

		local targetPolicyFunction = targetPolicyFunctionList[NewDeepOffPolicyMonteCarloControlModel.targetPolicyFunction]

		local discountFactor = NewDeepOffPolicyMonteCarloControlModel.discountFactor

		local numberOfActions = #actionVectorHistory[1]

		local outputDimensionSizeArray = {1, numberOfActions}

		local cVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 0) 

		local weightVector = AqwamTensorLibrary:createTensor(outputDimensionSizeArray, 1)

		local discountedReward = 0

		for h = #actionVectorHistory, 1, -1 do

			discountedReward = rewardValueHistory[h] + (discountFactor * discountedReward)

			cVector = AqwamTensorLibrary:add(cVector, weightVector)

			local actionVector = actionVectorHistory[h]

			local lossVectorPart1 = AqwamTensorLibrary:divide(weightVector, cVector)

			local lossVectorPart2 = AqwamTensorLibrary:subtract(discountedReward, actionVector)

			local lossVector = AqwamTensorLibrary:multiply(lossVectorPart1, lossVectorPart2, -1) -- The original non-deep off-policy Monte-Carlo Control version performs gradient ascent. But the neural network performs gradient descent. So, we need to negate the loss vector by multiplying it with -1 to make the neural network to perform gradient ascent.

			local targetActionVector = targetPolicyFunction(actionVector)

			local actionRatioVector = AqwamTensorLibrary:divide(targetActionVector, actionVector)

			weightVector = AqwamTensorLibrary:multiply(weightVector, actionRatioVector)

			Model:forwardPropagate(featureVectorHistory[h], true)

			Model:update(lossVector, true)

		end

		table.clear(featureVectorHistory)

		table.clear(actionVectorHistory)

		table.clear(rewardValueHistory)

	end)

	NewDeepOffPolicyMonteCarloControlModel:setResetFunction(function()

		table.clear(featureVectorHistory)

		table.clear(actionVectorHistory)

		table.clear(rewardValueHistory)

	end)

	return NewDeepOffPolicyMonteCarloControlModel

end

return DeepOffPolicyMonteCarloControlModel
