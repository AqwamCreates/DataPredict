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

	local meanVector = AqwamMatrixLibrary:horizontalMean(vector)

	local standardDeviationVector = AqwamMatrixLibrary:horizontalStandardDeviation(vector)

	local zScoreVectorPart1 = AqwamMatrixLibrary:subtract(vector, meanVector)

	local zScoreVector = AqwamMatrixLibrary:divide(zScoreVectorPart1, standardDeviationVector)

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

	local actionProbabilityValueHistory = {}
	
	NewAdvantageActorCriticModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local ActorModel = NewAdvantageActorCriticModel.ActorModel
		
		local CriticModel = NewAdvantageActorCriticModel.CriticModel
		
		local actionVector = ActorModel:predict(previousFeatureVector, true)
		
		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local actionProbabilityVector = calculateProbability(actionVector)
		
		local advantageValue = rewardValue + (NewAdvantageActorCriticModel.discountFactor * currentCriticValue) - previousCriticValue

		local actionIndex = table.find(ActorModel:getClassesList(), action)

		local actionProbabilityValue = actionProbabilityVector[1][actionIndex]
		
		local logActionProbabilityValue = math.log(actionProbabilityValue)

		table.insert(advantageValueHistory, advantageValue)

		table.insert(actionProbabilityValueHistory, logActionProbabilityValue)
		
		return advantageValue

	end)
	
	NewAdvantageActorCriticModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionVector, rewardValue, currentFeatureVector)

		local CriticModel = NewAdvantageActorCriticModel.CriticModel

		local zScoreVector, standardDeviationVector = AqwamMatrixLibrary:horizontalZScoreNormalization(actionVector)

		local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)

		local logStandardDeviationVector = AqwamMatrixLibrary:logarithm(standardDeviationVector)

		local multipliedLogStandardDeviationVector = AqwamMatrixLibrary:multiply(2, logStandardDeviationVector)

		local numberOfActionDimensions = #NewAdvantageActorCriticModel.ActorModel:getClassesList()

		local actionProbabilityValuePart1 = AqwamMatrixLibrary:sum(multipliedLogStandardDeviationVector)

		local actionProbabilityValue = -0.5 * (actionProbabilityValuePart1 + (numberOfActionDimensions * math.log(2 * math.pi)))

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewAdvantageActorCriticModel.discountFactor * currentCriticValue) - previousCriticValue

		table.insert(advantageValueHistory, advantageValue)

		table.insert(actionProbabilityValueHistory, actionProbabilityValue)
		
		return advantageValue

	end)

	NewAdvantageActorCriticModel:setEpisodeUpdateFunction(function()

		local historyLength = #advantageValueHistory

		local sumActorLoss = 0

		local sumCriticLoss = 0

		for h = 1, historyLength, 1 do

			local advantageValue = advantageValueHistory[h]

			local actorLoss = actionProbabilityValueHistory[h] * advantageValue

			sumCriticLoss = sumCriticLoss + advantageValue
			
			sumActorLoss = sumActorLoss + actorLoss

		end
		
		local ActorModel = NewAdvantageActorCriticModel.ActorModel

		local CriticModel = NewAdvantageActorCriticModel.CriticModel

		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local numberOfActions = #ActorModel:getClassesList()

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)
		local sumActorLossVector = AqwamMatrixLibrary:createMatrix(1, numberOfActions, -sumActorLoss)

		ActorModel:forwardPropagate(featureVector, true)
		CriticModel:forwardPropagate(featureVector, true)

		ActorModel:backwardPropagate(sumActorLossVector, true)
		CriticModel:backwardPropagate(-sumCriticLoss, true)

		table.clear(advantageValueHistory)

		table.clear(actionProbabilityValueHistory)

	end)

	NewAdvantageActorCriticModel:setResetFunction(function()

		table.clear(advantageValueHistory)

		table.clear(actionProbabilityValueHistory)

	end)
	
	return NewAdvantageActorCriticModel

end

return AdvantageActorCriticModel