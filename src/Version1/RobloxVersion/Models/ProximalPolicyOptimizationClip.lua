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

local ReinforcementLearningActorCriticBaseModel = require(script.Parent.ReinforcementLearningActorCriticBaseModel)

ProximalPolicyOptimizationClipModel = {}

ProximalPolicyOptimizationClipModel.__index = ProximalPolicyOptimizationClipModel

setmetatable(ProximalPolicyOptimizationClipModel, ReinforcementLearningActorCriticBaseModel)

local defaultClipRatio = 0.3

local function calculateProbability(vector)

	local zScoreVector, standardDeviationVector = AqwamMatrixLibrary:horizontalZScoreNormalization(vector)

	local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)

	local probabilityVectorPart1 = AqwamMatrixLibrary:multiply(-0.5, squaredZScoreVector)

	local probabilityVectorPart2 = AqwamMatrixLibrary:exponent(probabilityVectorPart1)

	local probabilityVectorPart3 = AqwamMatrixLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local probabilityVector = AqwamMatrixLibrary:divide(probabilityVectorPart2, probabilityVectorPart3)

	return probabilityVector

end

local function calculateRewardToGo(rewardHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward = rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function ProximalPolicyOptimizationClipModel.new(clipRatio, discountFactor)
	
	local NewProximalPolicyOptimizationClipModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)
	
	setmetatable(NewProximalPolicyOptimizationClipModel, ProximalPolicyOptimizationClipModel)
	
	NewProximalPolicyOptimizationClipModel.clipRatio = clipRatio or defaultClipRatio
	
	local rewardValueHistory = {}
	
	local criticValueHistory = {}
	
	local actionProbabilityVectorHistory = {}

	local oldActionProbabilityVectorHistory = {}
	
	local advantageValueHistory = {}
	
	local oldAdvantageValueHistory = {}
	
	local clipFunction = function(value) 

		local clipRatio = NewProximalPolicyOptimizationClipModel.clipRatio 

		return math.clamp(value, 1 - clipRatio, 1 + clipRatio) 

	end
	
	NewProximalPolicyOptimizationClipModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel

		local actionVector = NewProximalPolicyOptimizationClipModel.ActorModel:forwardPropagate(previousFeatureVector)

		local actionProbabilityVector = calculateProbability(actionVector)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationClipModel.discountFactor * currentCriticValue) - previousCriticValue

		local logActionProbabilityVector = AqwamMatrixLibrary:logarithm(actionProbabilityVector)

		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)

		table.insert(rewardValueHistory, rewardValue)

		return advantageValue

	end)

	NewProximalPolicyOptimizationClipModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, rewardValue, currentFeatureVector)

		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel

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

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationClipModel.discountFactor * currentCriticValue) - previousCriticValue
		
		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(advantageValueHistory, advantageValue)

		table.insert(rewardValueHistory, rewardValue)

		return advantageValue

	end)
	
	NewProximalPolicyOptimizationClipModel:setEpisodeUpdateFunction(function()
		
		local ActorModel = NewProximalPolicyOptimizationClipModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel
		
		if (#oldActionProbabilityVectorHistory == 0) then 
			
			oldActionProbabilityVectorHistory = table.clone(actionProbabilityVectorHistory)
			
			oldAdvantageValueHistory = table.clone(advantageValueHistory)
			
			table.clear(actionProbabilityVectorHistory)

			table.clear(criticValueHistory)

			table.clear(advantageValueHistory)

			table.clear(rewardValueHistory)
			
			return nil
				
		end

		if (#actionProbabilityVectorHistory ~= #oldActionProbabilityVectorHistory) then error("The number of updates does not equal to the number of old updates!") end
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewProximalPolicyOptimizationClipModel.discountFactor)

		local historyLength = #criticValueHistory
		
		local sumActorLossVector = AqwamMatrixLibrary:createMatrix(1, #actionProbabilityVectorHistory[1], 0)
		
		local sumCriticLoss = 0
		
		for h = 1, historyLength, 1 do

			local ratioVector = AqwamMatrixLibrary:divide(actionProbabilityVectorHistory[h], oldActionProbabilityVectorHistory[h])
			
			local oldAdvantageValue = oldAdvantageValueHistory[h]

			local actorLossVectorPart1 = AqwamMatrixLibrary:multiply(ratioVector, oldAdvantageValue)
			
			local clippedRatioVector = AqwamMatrixLibrary:applyFunction(clipFunction, ratioVector)
			
			local actorLossVectorPart2 = AqwamMatrixLibrary:multiply(clippedRatioVector, oldAdvantageValue)
			
			local actorLossVector = AqwamMatrixLibrary:applyFunction(math.min, actorLossVectorPart1, actorLossVectorPart2)

			local criticLoss = criticValueHistory[h] - rewardToGoArray[h]

			sumActorLossVector = AqwamMatrixLibrary:add(sumActorLossVector, actorLossVector)

			sumCriticLoss = sumCriticLoss + criticLoss

		end

		sumActorLossVector = AqwamMatrixLibrary:divide(sumActorLossVector, -historyLength)

		sumCriticLoss = sumCriticLoss / historyLength

		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

		ActorModel:forwardPropagate(featureVector, true, true)

		CriticModel:forwardPropagate(featureVector, true, true)

		ActorModel:backwardPropagate(sumActorLossVector, true)

		CriticModel:backwardPropagate(sumCriticLoss, true)

		oldActionProbabilityVectorHistory = table.clone(actionProbabilityVectorHistory)

		oldAdvantageValueHistory = table.clone(advantageValueHistory)

		table.clear(actionProbabilityVectorHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)

		table.clear(rewardValueHistory)
		
	end)
	
	NewProximalPolicyOptimizationClipModel:setResetFunction(function()
		
		table.clear(actionProbabilityVectorHistory)

		table.clear(oldActionProbabilityVectorHistory)

		table.clear(criticValueHistory)

		table.clear(advantageValueHistory)

		table.clear(oldAdvantageValueHistory)

		table.clear(rewardValueHistory)
		
	end)
	
	return NewProximalPolicyOptimizationClipModel
	
end

function ProximalPolicyOptimizationClipModel:setParameters(clipRatio, discountFactor)
	
	self.clipRatio = clipRatio or self.clipRatio

	self.discountFactor =  discountFactor or self.discountFactor
	
end

return ProximalPolicyOptimizationClipModel