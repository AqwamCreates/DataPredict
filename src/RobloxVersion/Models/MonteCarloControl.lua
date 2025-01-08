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

MonteCarloControlModel = {}

MonteCarloControlModel.__index = MonteCarloControlModel

setmetatable(MonteCarloControlModel, ReinforcementLearningBaseModel)

local function calculateProbability(vector)

	local zScoreVector, standardDeviationVector = AqwamMatrixLibrary:horizontalZScoreNormalization(vector)

	local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)

	local probabilityVectorPart1 = AqwamMatrixLibrary:multiply(-0.5, squaredZScoreVector)

	local probabilityVectorPart2 = AqwamMatrixLibrary:exponent(probabilityVectorPart1)

	local probabilityVectorPart3 = AqwamMatrixLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local probabilityVector = AqwamMatrixLibrary:divide(probabilityVectorPart2, probabilityVectorPart3)

	return probabilityVector

end

local function calculateRewardToGo(rewardValueHistory, discountFactor)

	local rewardToGoArray = {}

	local discountedReward = 0

	for h = #rewardValueHistory, 1, -1 do

		discountedReward = rewardValueHistory[h] + (discountFactor * discountedReward)

		table.insert(rewardToGoArray, 1, discountedReward)

	end

	return rewardToGoArray

end

function MonteCarloControlModel.new(discountFactor)

	local NewMonteCarloControlModel = ReinforcementLearningBaseModel.new(discountFactor)
	
	setmetatable(NewMonteCarloControlModel, MonteCarloControlModel)
	
	local actionProbabilityVectorHistory = {}
	
	local rewardValueHistory = {}
	
	NewMonteCarloControlModel:setCategoricalUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local actionVector = NewMonteCarloControlModel.Model:forwardPropagate(previousFeatureVector)
		
		local actionProbabilityVector = calculateProbability(actionVector)

		table.insert(actionProbabilityVectorHistory, actionProbabilityVector)
		
		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewMonteCarloControlModel:setDiagonalGaussianUpdateFunction(function(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, rewardValue, currentFeatureVector)

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

		table.insert(actionProbabilityVectorHistory, logActionProbabilityVector)

		table.insert(rewardValueHistory, rewardValue)

	end)
	
	NewMonteCarloControlModel:setEpisodeUpdateFunction(function()
		
		local Model = NewMonteCarloControlModel.Model
		
		local rewardToGoArray = calculateRewardToGo(rewardValueHistory, NewMonteCarloControlModel.discountFactor)
		
		local sumLossVector = AqwamMatrixLibrary:createMatrix(1, #actionProbabilityVectorHistory[1], 0)
		
		for h, actionProbabilityVector in ipairs(actionProbabilityVectorHistory) do
			
			local lossVector = AqwamMatrixLibrary:subtract(rewardToGoArray[h], actionProbabilityVector)

			sumLossVector = AqwamMatrixLibrary:add(sumLossVector, lossVector)
			
		end	
		
		local numberOfFeatures = Model:getTotalNumberOfNeurons(1)

		local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

		local meanLossVector = AqwamMatrixLibrary:divide(sumLossVector, #rewardValueHistory)
		
		Model:forwardPropagate(featureVector, true, true)

		Model:backwardPropagate(meanLossVector, true)
		
		table.clear(actionProbabilityVectorHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	NewMonteCarloControlModel:setResetFunction(function()

		table.clear(actionProbabilityVectorHistory)
		
		table.clear(rewardValueHistory)
		
	end)
	
	return NewMonteCarloControlModel

end

function MonteCarloControlModel:setParameters(discountFactor)

	self.discountFactor = discountFactor or self.discountFactor

end

return MonteCarloControlModel