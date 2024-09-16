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

AdvantageActorCriticModel = {}

AdvantageActorCriticModel.__index = AdvantageActorCriticModel

setmetatable(AdvantageActorCriticModel, ReinforcementLearningActorCriticBaseModel)

local function calculateProbability(vector)

	local zScoreVector, standardDeviationVector = AqwamMatrixLibrary:horizontalZScoreNormalization(vector)

	local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)

	local probabilityVectorPart1 = AqwamMatrixLibrary:multiply(-0.5, squaredZScoreVector)

	local probabilityVectorPart2 = AqwamMatrixLibrary:exponent(probabilityVectorPart1)

	local probabilityVectorPart3 = AqwamMatrixLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local probabilityVector = AqwamMatrixLibrary:divide(probabilityVectorPart2, probabilityVectorPart3)

	return probabilityVector

end

function AdvantageActorCriticModel.new(discountFactor)

	local NewAdvantageActorCriticModel = ReinforcementLearningActorCriticBaseModel.new(discountFactor)

	setmetatable(NewAdvantageActorCriticModel, AdvantageActorCriticModel)
	
	local advantageValueHistory = {}

	local actionProbabilityVectorHistory = {}
	
	NewAdvantageActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local CriticModel = NewAdvantageActorCriticModel.CriticModel
		
		local actionVector = NewAdvantageActorCriticModel.ActorModel:forwardPropagate(previousFeatureVector)
		
		local previousCriticValue = CriticModel:forwardPropagate(previousFeatureVector)[1][1]

		local currentCriticValue = CriticModel:forwardPropagate(currentFeatureVector)[1][1]

		local actionProbabilityVector = calculateProbability(actionVector)
		
		local advantageValue = rewardValue + (NewAdvantageActorCriticModel.discountFactor * currentCriticValue) - previousCriticValue

		local logActionProbabilityVector = AqwamMatrixLibrary:logarithm(actionProbabilityVector)
		
		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(advantageValueHistory, advantageValue)

		return advantageValue

	end)
	
	NewAdvantageActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, rewardValue, currentFeatureVector)

		local CriticModel = NewAdvantageActorCriticModel.CriticModel

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

		local advantageValue = rewardValue + (NewAdvantageActorCriticModel.discountFactor * currentCriticValue) - previousCriticValue
		
		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(advantageValueHistory, advantageValue)
		
		return advantageValue

	end)

	NewAdvantageActorCriticModel:setEpisodeUpdateFunction(function()

		local historyLength = #advantageValueHistory

		local sumActorLossVector = AqwamMatrixLibrary:createMatrix(1, #actionProbabilityVectorHistory[1], 0)

		local sumCriticLoss = 0

		for h = 1, historyLength, 1 do

			local advantageValue = advantageValueHistory[h]

			local actorLossVector = AqwamMatrixLibrary:multiply(actionProbabilityVectorHistory[h], advantageValue)

			sumCriticLoss = sumCriticLoss + advantageValue
			
			sumActorLossVector = AqwamMatrixLibrary:add(sumActorLossVector, actorLossVector)

		end
		
		local ActorModel = NewAdvantageActorCriticModel.ActorModel

		local CriticModel = NewAdvantageActorCriticModel.CriticModel

		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)
		
		sumActorLossVector = AqwamMatrixLibrary:unaryMinus(sumActorLossVector)

		ActorModel:forwardPropagate(featureVector, true, true)
		
		CriticModel:forwardPropagate(featureVector, true, true)

		ActorModel:backwardPropagate(sumActorLossVector, true)
		
		CriticModel:backwardPropagate(-sumCriticLoss, true)
		
		table.clear(actionProbabilityVectorHistory)

		table.clear(advantageValueHistory)

	end)

	NewAdvantageActorCriticModel:setResetFunction(function()
		
		table.clear(actionProbabilityVectorHistory)

		table.clear(advantageValueHistory)

	end)
	
	return NewAdvantageActorCriticModel

end

return AdvantageActorCriticModel