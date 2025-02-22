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

OffPolicyMonteCarloControlModel = {}

OffPolicyMonteCarloControlModel.__index = OffPolicyMonteCarloControlModel

setmetatable(OffPolicyMonteCarloControlModel, ReinforcementLearningBaseModel)

local defaultTargetPolicyFunction = "StableSoftmax"

local targetPolicyFunctionList = {
	
	["Greedy"] = function (actionVector)
		
		local targetActionVector = AqwamMatrixLibrary:createMatrix(1, #actionVector[1], 0)
		
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
	
	["Softmax"] = function (actionVector) -- apparently roblox doesn't really handle very small values such as math.exp(-1000), so I added a more stable computation exp(a) / exp(b) -> exp (a - b)

		local exponentActionVector = AqwamMatrixLibrary:applyFunction(math.exp, actionVector)

		local exponentActionSumVector = AqwamMatrixLibrary:horizontalSum(exponentActionVector)

		local targetActionVector = AqwamMatrixLibrary:divide(exponentActionVector, exponentActionSumVector)

		return targetActionVector

	end,

	["StableSoftmax"] = function (actionVector)
		
		local highestActionValue = AqwamMatrixLibrary:findMaximumValue(actionVector)

		local subtractedZVector = AqwamMatrixLibrary:subtract(actionVector, highestActionValue)

		local exponentActionVector = AqwamMatrixLibrary:applyFunction(math.exp, subtractedZVector)

		local exponentActionSumVector = AqwamMatrixLibrary:horizontalSum(exponentActionVector)

		local targetActionVector = AqwamMatrixLibrary:divide(exponentActionVector, exponentActionSumVector)

		return targetActionVector

	end,
	
}

function OffPolicyMonteCarloControlModel.new(targetPolicyFunction, discountFactor)

	local NewOffPolicyMonteCarloControlModel = ReinforcementLearningBaseModel.new(discountFactor)
	
	setmetatable(NewOffPolicyMonteCarloControlModel, OffPolicyMonteCarloControlModel)
	
	NewOffPolicyMonteCarloControlModel.targetPolicyFunction = targetPolicyFunction or defaultTargetPolicyFunction
	
	local featureVectorHistory = {}
	
	local actionVectorHistory = {}
	
	local rewardValueHistory = {}
	
	NewOffPolicyMonteCarloControlModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local actionVector = NewOffPolicyMonteCarloControlModel.Model:forwardPropagate(previousFeatureVector)
		
		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(actionVectorHistory, actionVector)
		
		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewOffPolicyMonteCarloControlModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, rewardValue, currentFeatureVector)

		local randomNormalVector = AqwamMatrixLibrary:createRandomNormalMatrix(1, #actionMeanVector[1])

		local actionVectorPart1 = AqwamMatrixLibrary:multiply(actionStandardDeviationVector, randomNormalVector)

		local actionVector = AqwamMatrixLibrary:add(actionMeanVector, actionVectorPart1)

		local zScoreVectorPart1 = AqwamMatrixLibrary:subtract(actionVector, actionMeanVector)

		local zScoreVector = AqwamMatrixLibrary:divide(zScoreVectorPart1, actionStandardDeviationVector)

		local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)

		local logActionProbabilityVectorPart1 = AqwamMatrixLibrary:logarithm(actionStandardDeviationVector)

		local logActionProbabilityVectorPart2 = AqwamMatrixLibrary:multiply(2, logActionProbabilityVectorPart1)

		local logActionProbabilityVectorPart3 = AqwamMatrixLibrary:add(squaredZScoreVector, logActionProbabilityVectorPart2)

		local logActionProbabilityVector = AqwamMatrixLibrary:add(logActionProbabilityVectorPart3, math.log(2 * math.pi))
		
		table.insert(featureVectorHistory, previousFeatureVector)

		table.insert(actionVectorHistory, logActionProbabilityVector)

		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewOffPolicyMonteCarloControlModel:setEpisodeUpdateFunction(function()
		
		local Model = NewOffPolicyMonteCarloControlModel.Model
		
		local targetPolicyFunction = targetPolicyFunctionList[NewOffPolicyMonteCarloControlModel.targetPolicyFunction]
		
		local discountFactor = NewOffPolicyMonteCarloControlModel.discountFactor
		
		local numberOfActions = #actionVectorHistory[1]
		
		local cVector = AqwamMatrixLibrary:createMatrix(1, numberOfActions, 0) 
		
		local weightVector = AqwamMatrixLibrary:createMatrix(1, numberOfActions, 1)
		
		local discountedReward = 0
		
		for h = #actionVectorHistory, 1, -1 do
			
			discountedReward = rewardValueHistory[h] + (discountFactor * discountedReward)
			
			cVector = AqwamMatrixLibrary:add(cVector, weightVector)
			
			local actionVector = actionVectorHistory[h]
			
			local lossVectorPart1 = AqwamMatrixLibrary:divide(weightVector, cVector)
			
			local lossVectorPart2 = AqwamMatrixLibrary:subtract(discountedReward, actionVector)
			
			local lossVector = AqwamMatrixLibrary:multiply(lossVectorPart1, lossVectorPart2)
			
			local targetActionVector = targetPolicyFunction(actionVector)
			
			local actionRatioVector = AqwamMatrixLibrary:divide(targetActionVector, actionVector)
			
			weightVector = AqwamMatrixLibrary:multiply(weightVector, actionRatioVector)
			
			Model:forwardPropagate(featureVectorHistory[h], true, true)
			
			Model:backwardPropagate(lossVector, true)
			
		end
		
		table.clear(featureVectorHistory)
		
		table.clear(actionVectorHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	NewOffPolicyMonteCarloControlModel:setResetFunction(function()
		
		table.clear(featureVectorHistory)

		table.clear(actionVectorHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	return NewOffPolicyMonteCarloControlModel

end

function OffPolicyMonteCarloControlModel:setParameters(targetPolicyFunction, discountFactor)
	
	self.targetPolicyFunction = targetPolicyFunction or self.targetPolicyFunction

	self.discountFactor = discountFactor or self.discountFactor

end

return OffPolicyMonteCarloControlModel