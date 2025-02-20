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

VanillaPolicyGradientModel = {}

VanillaPolicyGradientModel.__index = VanillaPolicyGradientModel

setmetatable(VanillaPolicyGradientModel, ReinforcementLearningActorCriticBaseModel)

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

function VanillaPolicyGradientModel.new(discountFactor)
	
	local NewVanillaPolicyGradientModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)

	setmetatable(NewVanillaPolicyGradientModel, VanillaPolicyGradientModel)
	
	local actorLossVectorHistory = {}
	
	local criticValueHistory = {}

	local rewardValueHistory = {}

	NewVanillaPolicyGradientModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local actionVector = NewVanillaPolicyGradientModel.ActorModel:forwardPropagate(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(actionVector)

		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local advantageValue = rewardValue + (NewVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue
		
		local logActionProbabilityVector = AqwamMatrixLibrary:logarithm(actionProbabilityVector)
		
		local actorLossVector = AqwamMatrixLibrary:multiply(logActionProbabilityVector, advantageValue)
		
		table.insert(criticValueHistory, previousCriticValue)
		
		table.insert(actorLossVectorHistory, actorLossVector)

		table.insert(rewardValueHistory, rewardValue)

		return advantageValue

	end)
	
	NewVanillaPolicyGradientModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, rewardValue, currentFeatureVector)

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

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

		local advantageValue = rewardValue + (NewVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue

		local actorLossVector = AqwamMatrixLibrary:multiply(logActionProbabilityVector, advantageValue)

		table.insert(criticValueHistory, currentCriticValue)
		
		table.insert(actorLossVectorHistory, actorLossVector)
		
		table.insert(rewardValueHistory, rewardValue)
		
		return advantageValue

	end)

	NewVanillaPolicyGradientModel:setEpisodeUpdateFunction(function()
		
		local historyLength = #rewardValueHistory
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewVanillaPolicyGradientModel.discountFactor)
		
		local sumActorLossVector = AqwamMatrixLibrary:createMatrix(1, #actorLossVectorHistory[1], 0)
		
		local sumCriticLoss = 0
		
		for h = 1, historyLength, 1 do
			
			sumActorLossVector = AqwamMatrixLibrary:add(sumActorLossVector, actorLossVectorHistory[h])
			
			sumCriticLoss = sumCriticLoss + (criticValueHistory[h] - rewardToGoArray[h])
			
		end
		
		sumCriticLoss = sumCriticLoss / historyLength
		
		local ActorModel = NewVanillaPolicyGradientModel.ActorModel

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)
		
		sumActorLossVector = AqwamMatrixLibrary:unaryMinus(sumActorLossVector)

		ActorModel:forwardPropagate(featureVector, true, true)
		
		CriticModel:forwardPropagate(featureVector, true, true)

		ActorModel:backwardPropagate(sumActorLossVector, true)
		
		CriticModel:backwardPropagate(sumCriticLoss, true)

		table.clear(criticValueHistory)
		
		table.clear(actorLossVectorHistory)

		table.clear(rewardValueHistory)

	end)

	NewVanillaPolicyGradientModel:setResetFunction(function()

		table.clear(criticValueHistory)
		
		table.clear(actorLossVectorHistory)

		table.clear(rewardValueHistory)

	end)
	
	return NewVanillaPolicyGradientModel
	
end

return VanillaPolicyGradientModel
