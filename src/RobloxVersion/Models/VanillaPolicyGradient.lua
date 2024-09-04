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
	
	local actorLossValueHistory = {}
	
	local criticValueHistory = {}

	local rewardValueHistory = {}

	NewVanillaPolicyGradientModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local ActorModel = NewVanillaPolicyGradientModel.ActorModel

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local actionVector = ActorModel:predict(previousFeatureVector, true)

		local actionProbabilityVector = calculateProbability(actionVector)

		local actionIndex = table.find(ActorModel:getClassesList(), action)

		local actionProbability = actionProbabilityVector[1][actionIndex]

		local logActionProbability = math.log(actionProbability)

		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue
		
		local actorLossValue = logActionProbability * advantageValue
		
		table.insert(actorLossValueHistory, actorLossValue)
		
		table.insert(criticValueHistory, currentCriticValue)

		table.insert(rewardValueHistory, rewardValue)

		return advantageValue

	end)

	NewVanillaPolicyGradientModel:setCategoricalEpisodeUpdateFunction(function()
		
		local historyLength = #rewardValueHistory
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewVanillaPolicyGradientModel.discountFactor)
		
		local sumActorLoss = 0
		
		local sumCriticLoss = 0
		
		for h = 1, historyLength, 1 do
			
			sumActorLoss = sumActorLoss + actorLossValueHistory[h]
			
			sumCriticLoss = sumCriticLoss + (criticValueHistory[h] - rewardToGoArray[h])
			
		end
		
		sumCriticLoss = sumCriticLoss / historyLength
		
		local ActorModel = NewVanillaPolicyGradientModel.ActorModel

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local numberOfLayers = ActorModel:getNumberOfLayers()

		local numberOfNeuronsAtFinalLayer = ActorModel:getTotalNumberOfNeurons(numberOfLayers)

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)
		local sumActorLossVector = AqwamMatrixLibrary:createMatrix(1, numberOfNeuronsAtFinalLayer, -sumActorLoss)

		ActorModel:forwardPropagate(featureVector, true)
		CriticModel:forwardPropagate(featureVector, true)

		ActorModel:backwardPropagate(sumActorLossVector, true)
		CriticModel:backwardPropagate(-sumCriticLoss, true)
		
		table.clear(actorLossValueHistory)

		table.clear(criticValueHistory)

		table.clear(rewardValueHistory)

	end)

	NewVanillaPolicyGradientModel:setCategoricalResetFunction(function()

		table.clear(actorLossValueHistory)

		table.clear(criticValueHistory)

		table.clear(rewardValueHistory)

	end)
	
	NewVanillaPolicyGradientModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionVector, rewardValue, currentFeatureVector)
		
		local ActorModel = NewVanillaPolicyGradientModel.ActorModel

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel
		
		local zScoreVector, standardDeviationVector = AqwamMatrixLibrary:horizontalZScoreNormalization(actionVector)

		local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)

		local logStandardDeviationVector = AqwamMatrixLibrary:logarithm(standardDeviationVector)

		local multipliedLogStandardDeviationVector = AqwamMatrixLibrary:multiply(2, logStandardDeviationVector)

		local numberOfActionDimensions = #NewVanillaPolicyGradientModel.Model:getClassesList()

		local logLikelihoodPart1 = AqwamMatrixLibrary:sum(multipliedLogStandardDeviationVector)

		local logLikelihood = -0.5 * (logLikelihoodPart1 + (numberOfActionDimensions * math.log(2 * math.pi)))
		
		local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

		local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

		local advantageValue = rewardValue + (NewVanillaPolicyGradientModel.discountFactor * currentCriticValue) - previousCriticValue

		local actorLossValue = logLikelihood * advantageValue

		table.insert(actorLossValueHistory, actorLossValue)

		table.insert(criticValueHistory, currentCriticValue)

		table.insert(rewardValueHistory, rewardValue)
		
	end)
	
	NewVanillaPolicyGradientModel:setDiagonalGaussianEpisodeUpdateFunction(function()

		local historyLength = #rewardValueHistory

		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewVanillaPolicyGradientModel.discountFactor)

		local sumActorLoss = 0

		local sumCriticLoss = 0

		for h = 1, historyLength, 1 do

			sumActorLoss = sumActorLoss + actorLossValueHistory[h]

			sumCriticLoss = sumCriticLoss + (criticValueHistory[h] - rewardToGoArray[h])

		end

		sumCriticLoss = sumCriticLoss / historyLength

		local ActorModel = NewVanillaPolicyGradientModel.ActorModel

		local CriticModel = NewVanillaPolicyGradientModel.CriticModel

		local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

		local numberOfLayers = ActorModel:getNumberOfLayers()

		local numberOfNeuronsAtFinalLayer = ActorModel:getTotalNumberOfNeurons(numberOfLayers)

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)
		local sumActorLossVector = AqwamMatrixLibrary:createMatrix(1, numberOfNeuronsAtFinalLayer, -sumActorLoss)

		ActorModel:forwardPropagate(featureVector, true)
		CriticModel:forwardPropagate(featureVector, true)

		ActorModel:backwardPropagate(sumActorLossVector, true)
		CriticModel:backwardPropagate(-sumCriticLoss, true)

		table.clear(actorLossValueHistory)

		table.clear(criticValueHistory)

		table.clear(rewardValueHistory)

	end)

	NewVanillaPolicyGradientModel:setDiagonalGaussianResetFunction(function()

		table.clear(actorLossValueHistory)

		table.clear(criticValueHistory)

		table.clear(rewardValueHistory)

	end)
	
	return NewVanillaPolicyGradientModel
	
end

return VanillaPolicyGradientModel