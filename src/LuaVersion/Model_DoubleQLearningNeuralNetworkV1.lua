--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local ReinforcementLearningNeuralNetworkBaseModel = require("Model_ReinforcementLearningNeuralNetworkBaseModel")

DoubleQLearningNeuralNetworkModel = {}

DoubleQLearningNeuralNetworkModel.__index = DoubleQLearningNeuralNetworkModel

setmetatable(DoubleQLearningNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

function DoubleQLearningNeuralNetworkModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	local NewDoubleQLearningNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	setmetatable(NewDoubleQLearningNeuralNetworkModel, DoubleQLearningNeuralNetworkModel)
	
	NewDoubleQLearningNeuralNetworkModel.ModelParametersArray = {}
	
	NewDoubleQLearningNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)
		
		local randomProbability = Random.new():NextNumber()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		NewDoubleQLearningNeuralNetworkModel:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		local lossVector, temporalDifferenceError = NewDoubleQLearningNeuralNetworkModel:generateLossVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

		NewDoubleQLearningNeuralNetworkModel:saveModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		NewDoubleQLearningNeuralNetworkModel:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)

		NewDoubleQLearningNeuralNetworkModel:forwardPropagate(previousFeatureVector, true)
		
		NewDoubleQLearningNeuralNetworkModel:backPropagate(lossVector, true)

		NewDoubleQLearningNeuralNetworkModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)
		
		return temporalDifferenceError
		
	end)

	return NewDoubleQLearningNeuralNetworkModel

end

function DoubleQLearningNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

function DoubleQLearningNeuralNetworkModel:saveModelParametersFromModelParametersArray(index)

	local ModelParameters = self:getModelParameters()

	self.ModelParametersArray[index] = ModelParameters

end

function DoubleQLearningNeuralNetworkModel:loadModelParametersFromModelParametersArray(index)
	
	local FirstModelParameters = self.ModelParametersArray[1]
	
	local SecondModelParameters = self.ModelParametersArray[2]
	
	if (FirstModelParameters == nil) and (SecondModelParameters == nil) then
		
		self:generateLayers()
		
		self:saveModelParametersFromModelParametersArray(1)
		
		self:saveModelParametersFromModelParametersArray(2)
		
	end
	
	local CurrentModelParameters = self.ModelParametersArray[index]
	
	self:setModelParameters(CurrentModelParameters, true)
	
end

function DoubleQLearningNeuralNetworkModel:generateLossVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local predictedValue, maxQValue = self:predict(currentFeatureVector)

	local targetValue = rewardValue + (self.discountFactor * maxQValue[1][1])
	
	local numberOfClasses = #self:getClassesList()

	local previousVector = self:predict(previousFeatureVector, true)

	local actionIndex = table.find(self.ClassesList, action)
	
	local lastValue = previousVector[1][actionIndex]
	
	local temporalDifferenceError = targetValue - lastValue
		
	local lossVector = AqwamMatrixLibrary:createMatrix(1, numberOfClasses, 0)

	lossVector[1][actionIndex] = temporalDifferenceError
	
	return lossVector, temporalDifferenceError
	
end

function DoubleQLearningNeuralNetworkModel:setModelParameters1(ModelParameters1)

	self.ModelParametersArray[1] = ModelParameters1

end

function DoubleQLearningNeuralNetworkModel:setModelParameters2(ModelParameters2)

	self.ModelParametersArray[2] = ModelParameters2

end

function DoubleQLearningNeuralNetworkModel:getModelParameters1(ModelParameters1)

	return self.ModelParametersArray[1]

end

function DoubleQLearningNeuralNetworkModel:getModelParameters2(ModelParameters2)

	return self.ModelParametersArray[2]

end

return DoubleQLearningNeuralNetworkModel
