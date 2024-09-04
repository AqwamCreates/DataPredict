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
	
	local actionProbabilityValueHistory = {}

	local oldActionProbabilityValueHistory = {}
	
	local advantageValueHistory = {}
	
	local oldAdvantageValueHistory = {}
	
	local clipFunction = function(value) 

		local clipRatio = NewProximalPolicyOptimizationClipModel.clipRatio 

		return math.clamp(value, 1 - clipRatio, 1 + clipRatio) 

	end
	
	NewProximalPolicyOptimizationClipModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local ActorModel = NewProximalPolicyOptimizationClipModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel

		local actionVector = ActorModel:predict(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(actionVector)

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationClipModel.discountFactor * currentCriticValue) - previousCriticValue

		local actionIndex = table.find(ActorModel:getClassesList(), action)

		local actionProbabilityValue = actionProbabilityVector[1][actionIndex]

		local logActionProbabilityValue = math.log(actionProbabilityValue)

		table.insert(advantageValueHistory, advantageValue)

		table.insert(criticValueHistory, currentCriticValue)

		table.insert(actionProbabilityValueHistory, logActionProbabilityValue)

		table.insert(rewardValueHistory, rewardValue)

		return advantageValue

	end)

	NewProximalPolicyOptimizationClipModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionVector, rewardValue, currentFeatureVector)

		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel

		local zScoreVector, standardDeviationVector = AqwamMatrixLibrary:horizontalZScoreNormalization(actionVector)

		local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)

		local logStandardDeviationVector = AqwamMatrixLibrary:logarithm(standardDeviationVector)

		local multipliedLogStandardDeviationVector = AqwamMatrixLibrary:multiply(2, logStandardDeviationVector)

		local numberOfActionDimensions = #NewProximalPolicyOptimizationClipModel.ActorModel:getClassesList()

		local actionProbabilityValuePart1 = AqwamMatrixLibrary:sum(multipliedLogStandardDeviationVector)

		local actionProbabilityValue = -0.5 * (actionProbabilityValuePart1 + (numberOfActionDimensions * math.log(2 * math.pi)))

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewProximalPolicyOptimizationClipModel.discountFactor * currentCriticValue) - previousCriticValue

		table.insert(advantageValueHistory, advantageValue)

		table.insert(criticValueHistory, currentCriticValue)

		table.insert(actionProbabilityValueHistory, actionProbabilityValue)

		table.insert(rewardValueHistory, rewardValue)

		return advantageValue

	end)
	
	NewProximalPolicyOptimizationClipModel:setEpisodeUpdateFunction(function()
		
		local ActorModel = NewProximalPolicyOptimizationClipModel.ActorModel

		local CriticModel = NewProximalPolicyOptimizationClipModel.CriticModel
		
		if (#oldActionProbabilityValueHistory == 0) then 
			
			oldActionProbabilityValueHistory = table.clone(actionProbabilityValueHistory)
			
			oldAdvantageValueHistory = table.clone(advantageValueHistory)
			
			table.clear(advantageValueHistory)

			table.clear(criticValueHistory)

			table.clear(rewardValueHistory)

			table.clear(actionProbabilityValueHistory)
			
			return 
				
		end
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewProximalPolicyOptimizationClipModel.discountFactor)

		local historyLength = #criticValueHistory
		
		local sumActorLoss = 0
		
		local sumCriticLoss = 0
		
		for h = 1, historyLength, 1 do

			local ratio = actionProbabilityValueHistory[h] / oldActionProbabilityValueHistory[h]
			
			local oldAdvantageValueHistory = oldAdvantageValueHistory[h]

			local actorSurrogateLossPart1 = ratio * oldAdvantageValueHistory
			
			local actorSurrogateLossPart2 = clipFunction(ratio) * oldAdvantageValueHistory
			
			local actorLoss = math.min(actorSurrogateLossPart1, actorSurrogateLossPart2)

			local criticLoss = criticValueHistory[h] - rewardToGoArray[h]

			sumActorLoss = sumActorLoss + actorLoss

			sumCriticLoss = sumCriticLoss + criticLoss

		end

		sumActorLoss = sumActorLoss / historyLength

		sumCriticLoss = sumCriticLoss / historyLength

		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local numberOfActions = #ActorModel:getClassesList()

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

		local sumActorLossVector = AqwamMatrixLibrary:createMatrix(1, numberOfActions, -sumActorLoss)

		ActorModel:forwardPropagate(featureVector, true)

		CriticModel:forwardPropagate(featureVector, true)

		ActorModel:backwardPropagate(sumActorLossVector, true)

		CriticModel:backwardPropagate(sumCriticLoss, true)

		oldActionProbabilityValueHistory = table.clone(actionProbabilityValueHistory)

		oldAdvantageValueHistory = table.clone(advantageValueHistory)

		table.clear(advantageValueHistory)

		table.clear(criticValueHistory)

		table.clear(rewardValueHistory)

		table.clear(actionProbabilityValueHistory)
		
	end)
	
	NewProximalPolicyOptimizationClipModel:setResetFunction(function()
		
		table.clear(advantageValueHistory)
		
		table.clear(oldAdvantageValueHistory)

		table.clear(criticValueHistory)

		table.clear(rewardValueHistory)

		table.clear(actionProbabilityValueHistory)
		
		table.clear(oldActionProbabilityValueHistory)
		
	end)
	
	return NewProximalPolicyOptimizationClipModel
	
end

function ProximalPolicyOptimizationClipModel:setParameters(clipRatio, discountFactor)
	
	self.clipRatio = clipRatio or self.clipRatio

	self.discountFactor =  discountFactor or self.discountFactor
	
end

return ProximalPolicyOptimizationClipModel