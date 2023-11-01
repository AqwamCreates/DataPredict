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
		- USED AS COMMERCIAL USE OR PUBLIC USE
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local ReinforcementLearningNeuralNetworkBaseModel = require("Model_ReinforcementLearningNeuralNetworkBaseModel")

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

DoubleStateActionRewardStateActionNeuralNetworkModel = {}

DoubleStateActionRewardStateActionNeuralNetworkModel.__index = DoubleStateActionRewardStateActionNeuralNetworkModel

setmetatable(DoubleStateActionRewardStateActionNeuralNetworkModel, ReinforcementLearningNeuralNetworkBaseModel)

function DoubleStateActionRewardStateActionNeuralNetworkModel.new(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	local NewDoubleStateActionRewardStateActionNeuralNetworkModel = ReinforcementLearningNeuralNetworkBaseModel.new(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	setmetatable(NewDoubleStateActionRewardStateActionNeuralNetworkModel, DoubleStateActionRewardStateActionNeuralNetworkModel)

	NewDoubleStateActionRewardStateActionNeuralNetworkModel.ModelParametersArray = {}

	NewDoubleStateActionRewardStateActionNeuralNetworkModel:setUpdateFunction(function(previousFeatureVector, action, rewardValue, currentFeatureVector)

		local randomProbability = Random.new():NextNumber()

		local updateSecondModel = (randomProbability >= 0.5)

		local selectedModelNumberForTargetVector = (updateSecondModel and 1) or 2

		local selectedModelNumberForUpdate = (updateSecondModel and 2) or 1

		NewDoubleStateActionRewardStateActionNeuralNetworkModel:loadModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		local targetVector = NewDoubleStateActionRewardStateActionNeuralNetworkModel:generateTargetVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

		NewDoubleStateActionRewardStateActionNeuralNetworkModel:saveModelParametersFromModelParametersArray(selectedModelNumberForTargetVector)

		NewDoubleStateActionRewardStateActionNeuralNetworkModel:loadModelParametersFromModelParametersArray(selectedModelNumberForUpdate)

		NewDoubleStateActionRewardStateActionNeuralNetworkModel:train(previousFeatureVector, targetVector)

		NewDoubleStateActionRewardStateActionNeuralNetworkModel:saveModelParametersFromModelParametersArray(selectedModelNumberForUpdate)

	end)

	return NewDoubleStateActionRewardStateActionNeuralNetworkModel

end

function DoubleStateActionRewardStateActionNeuralNetworkModel:setParameters(maxNumberOfIterations, learningRate, targetCost, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor)

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.learningRate = learningRate or self.learningRate

	self.targetCost = targetCost or self.targetCost

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.currentEpsilon = epsilon or self.currentEpsilon

end

function DoubleStateActionRewardStateActionNeuralNetworkModel:setModelParametersArray(ModelParameters1, ModelParameters2)

	if (ModelParameters1) or (ModelParameters2) then

		self.ModelParametersArray = {ModelParameters1, ModelParameters2}

	else

		self.ModelParametersArray = {}

	end

end

function DoubleStateActionRewardStateActionNeuralNetworkModel:getModelParametersArray()

	return self.ModelParametersArray

end

function DoubleStateActionRewardStateActionNeuralNetworkModel:saveModelParametersFromModelParametersArray(index)

	local ModelParameters = self:getModelParameters()

	self.ModelParametersArray[index] = ModelParameters

end

function DoubleStateActionRewardStateActionNeuralNetworkModel:loadModelParametersFromModelParametersArray(index)

	local FirstModelParameters = self.ModelParametersArray[1]

	local SecondModelParameters = self.ModelParametersArray[2]

	if (FirstModelParameters == nil) and (SecondModelParameters == nil) then

		self:generateLayers()

		self:saveModelParametersFromModelParametersArray(1)

		self:saveModelParametersFromModelParametersArray(2)

	end

	local CurrentModelParameters = self.ModelParametersArray[index]

	self:setModelParameters(CurrentModelParameters)

end

function DoubleStateActionRewardStateActionNeuralNetworkModel:generateTargetVector(previousFeatureVector, action, rewardValue, currentFeatureVector)

	local targetVector = self:predict(currentFeatureVector, true)

	local dicountedVector = AqwamMatrixLibrary:multiply(self.discountFactor, targetVector)

	local newTargetVector = AqwamMatrixLibrary:add(rewardValue, dicountedVector)

	return newTargetVector

end

return DoubleStateActionRewardStateActionNeuralNetworkModel

