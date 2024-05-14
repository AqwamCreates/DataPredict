local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

local ReinforcementLearningActorCriticBaseModel = require(script.Parent.Parent.Models.ReinforcementLearningActorCriticBaseModel)

WeightProximalPolicyOptimizationClipModel = {}

WeightProximalPolicyOptimizationClipModel.__index = WeightProximalPolicyOptimizationClipModel

setmetatable(WeightProximalPolicyOptimizationClipModel, ReinforcementLearningActorCriticBaseModel)

local defaultClipRatio = 0.3

local function calculateProbability(outputMatrix)

	local meanVector = AqwamMatrixLibrary:horizontalMean(outputMatrix)

	local standardDeviationVector = AqwamMatrixLibrary:horizontalStandardDeviation(outputMatrix)

	local zScoreVectorPart1 = AqwamMatrixLibrary:subtract(outputMatrix, meanVector)

	local zScoreVector = AqwamMatrixLibrary:divide(zScoreVectorPart1, standardDeviationVector)

	local zScoreSquaredVector = AqwamMatrixLibrary:power(zScoreVector, 2)

	local probabilityVectorPart1 = AqwamMatrixLibrary:multiply(-0.5, zScoreSquaredVector)

	local probabilityVectorPart2 = AqwamMatrixLibrary:applyFunction(math.exp, probabilityVectorPart1)

	local probabilityVectorPart3 = AqwamMatrixLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local probabilityVector = AqwamMatrixLibrary:divide(probabilityVectorPart2, probabilityVectorPart3)

	return probabilityVector

end

local function calculateRewardsToGo(rewardHistory, discountFactor)

	local rewardsToGoArray = {}

	local discountedReward = 0

	for h = #rewardHistory, 1, -1 do

		discountedReward = rewardHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardsToGoArray, 1, discountedReward)

	end

	return rewardsToGoArray

end

function WeightProximalPolicyOptimizationClipModel.new(clipRatio, discountFactor)
	
	local NewWeightProximalPolicyOptimizationClipModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)
	
	setmetatable(NewWeightProximalPolicyOptimizationClipModel, WeightProximalPolicyOptimizationClipModel)
	
	NewWeightProximalPolicyOptimizationClipModel.clipRatio = clipRatio or defaultClipRatio
	
	local rewardHistory = {}
	
	local criticValueHistory = {}
	
	local actionVectorHistory = {}
	
	local advantageValueHistory = {}
	
	local oldAdvantageValueHistory = {}
	
	local OldActorModelParameters
	
	local OldCriticModelParameters
	
	NewWeightProximalPolicyOptimizationClipModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local ActorModel = NewWeightProximalPolicyOptimizationClipModel.ActorModel
		
		local CriticModel = NewWeightProximalPolicyOptimizationClipModel.CriticModel
		
		if (ActorModel:getModelParameters() == nil) then CriticModel:generateLayers() end
		
		if (CriticModel:getModelParameters() == nil) then CriticModel:generateLayers() end
		
		local allOutputsMatrix = ActorModel:predict(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(allOutputsMatrix)

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewWeightProximalPolicyOptimizationClipModel.discountFactor * currentCriticValue) - previousCriticValue

		table.insert(advantageValueHistory, advantageValue)

		table.insert(criticValueHistory, previousCriticValue)

		table.insert(actionVectorHistory, actionProbabilityVector)
		
		table.insert(rewardHistory, rewardValue)
		
	end)
	
	NewWeightProximalPolicyOptimizationClipModel:setEpisodeUpdateFunction(function()
		
		local ActorModel = NewWeightProximalPolicyOptimizationClipModel.ActorModel

		local CriticModel = NewWeightProximalPolicyOptimizationClipModel.CriticModel
		
		if (OldActorModelParameters == nil)  or (OldCriticModelParameters == nil) then 
			
			OldActorModelParameters = ActorModel:getModelParameters(false)
			
			OldCriticModelParameters = CriticModel:getModelParameters(false)
			
			ActorModel:setModelParameters(nil)
			
			CriticModel:setModelParameters(nil)

			oldAdvantageValueHistory = table.clone(advantageValueHistory)

			table.clear(advantageValueHistory)

			table.clear(criticValueHistory)

			table.clear(rewardHistory)
			
			return 
				
		end
		
		local rewardsToGoArray = calculateRewardsToGo(rewardHistory, NewWeightProximalPolicyOptimizationClipModel.discountFactor)

		local historyLength = #criticValueHistory
		
		local sumAdvantageValue = 0

		local sumCriticLoss = 0
		
		local clipFunction = function(value) 
			
			local clipRatio = NewWeightProximalPolicyOptimizationClipModel.clipRatio 
			
			return math.clamp(value, 1 - clipRatio, 1 + clipRatio) 
			
		end

		for h = 1, historyLength, 1 do
			
			local advantageValue = advantageValueHistory[h]

			local criticLoss = math.pow(rewardsToGoArray[h] - criticValueHistory[h], 2)

			sumAdvantageValue = sumAdvantageValue + advantageValue

			sumCriticLoss = sumCriticLoss + criticLoss

		end
		
		local NewActorModelParameters = {}
		
		local NewCriticModelParameters = {}
		
		for i, weightMatrix in ipairs(ActorModel:getModelParameters(false)) do
			
			local weightRatio = AqwamMatrixLibrary:divide(weightMatrix, OldActorModelParameters[i])
			
			weightRatio = AqwamMatrixLibrary:applyFunction(clipFunction, weightRatio)
			
			NewActorModelParameters[i] = AqwamMatrixLibrary:multiply(weightRatio, sumAdvantageValue)
			
		end
		
		for i, weightMatrix in ipairs(CriticModel:getModelParameters(false)) do

			local weightRatio = AqwamMatrixLibrary:divide(weightMatrix, OldActorModelParameters[i])
			
			weightRatio = AqwamMatrixLibrary:applyFunction(clipFunction, weightRatio)
			
			NewCriticModelParameters[i] = AqwamMatrixLibrary:multiply(weightRatio, sumCriticLoss)

		end
		
		oldAdvantageValueHistory = table.clone(advantageValueHistory)
		
		table.clear(advantageValueHistory)

		table.clear(criticValueHistory)

		table.clear(rewardHistory)
		
	end)
	
	NewWeightProximalPolicyOptimizationClipModel:extendResetFunction(function()
		
		table.clear(advantageValueHistory)

		table.clear(oldAdvantageValueHistory)

		table.clear(criticValueHistory)

		table.clear(rewardHistory)
		
	end)
	
	return NewWeightProximalPolicyOptimizationClipModel
	
end

function WeightProximalPolicyOptimizationClipModel:setParameters(clipRatio, discountFactor)
	
	self.clipRatio = clipRatio or self.clipRatio

	self.discountFactor =  discountFactor or self.discountFactor
	
end

return WeightProximalPolicyOptimizationClipModel
