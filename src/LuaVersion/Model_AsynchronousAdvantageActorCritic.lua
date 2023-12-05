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

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

AsynchronousAdvantageActorCriticModel = {}

AsynchronousAdvantageActorCriticModel.__index = AsynchronousAdvantageActorCriticModel

local defaultLearningRate = 0.1

local defaultNumberOfReinforcementsPerEpisode = 10

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultRewardAveragingRate = 0.05 -- The higher the value, the higher the episodic reward, but lower the running reward.

local defaultTotalNumberOfReinforcementsToUpdateMainModel = 100

function AsynchronousAdvantageActorCriticModel.new(learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, rewardAveragingRate, totalNumberOfReinforcementsToUpdateMainModel)
	
	local NewAsynchronousAdvantageActorCriticModel = {}
	
	setmetatable(NewAsynchronousAdvantageActorCriticModel, AsynchronousAdvantageActorCriticModel)
	
	NewAsynchronousAdvantageActorCriticModel.learningRate = learningRate or defaultLearningRate
	
	NewAsynchronousAdvantageActorCriticModel.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewAsynchronousAdvantageActorCriticModel.epsilon = epsilon or defaultEpsilon

	NewAsynchronousAdvantageActorCriticModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewAsynchronousAdvantageActorCriticModel.discountFactor =  discountFactor or defaultDiscountFactor
	
	NewAsynchronousAdvantageActorCriticModel.rewardAveragingRate = rewardAveragingRate or defaultRewardAveragingRate
	
	NewAsynchronousAdvantageActorCriticModel.currentEpsilonArray = {}

	NewAsynchronousAdvantageActorCriticModel.previousFeatureVectorArray = {}

	NewAsynchronousAdvantageActorCriticModel.printReinforcementOutput = true

	NewAsynchronousAdvantageActorCriticModel.currentNumberOfReinforcementsArray = {}

	NewAsynchronousAdvantageActorCriticModel.currentNumberOfEpisodesArray = {}
	
	NewAsynchronousAdvantageActorCriticModel.advantageHistoryArray = {}
	
	NewAsynchronousAdvantageActorCriticModel.actionProbabilityHistoryArray = {}
	
	NewAsynchronousAdvantageActorCriticModel.criticValueHistoryArray = {}
	
	NewAsynchronousAdvantageActorCriticModel.episodeRewardArray = {}
	
	NewAsynchronousAdvantageActorCriticModel.runningRewardArray = {}
	
	NewAsynchronousAdvantageActorCriticModel.ActorModelArray = {}
	
	NewAsynchronousAdvantageActorCriticModel.CriticModelArray = {}
	
	NewAsynchronousAdvantageActorCriticModel.ActorModelCostFunctionDerivativesArray = {}

	NewAsynchronousAdvantageActorCriticModel.CriticModelCostFunctionDerivativesArray = {}
	
	NewAsynchronousAdvantageActorCriticModel.ExperienceReplayArray = {}
	
	NewAsynchronousAdvantageActorCriticModel.ClassesList = nil
	
	NewAsynchronousAdvantageActorCriticModel.totalNumberOfReinforcementsToUpdateMainModel = totalNumberOfReinforcementsToUpdateMainModel or defaultTotalNumberOfReinforcementsToUpdateMainModel
	
	NewAsynchronousAdvantageActorCriticModel.currentTotalNumberOfReinforcementsToUpdateMainModel = 0
	
	NewAsynchronousAdvantageActorCriticModel.ActorMainModelParameters = nil
	
	NewAsynchronousAdvantageActorCriticModel.CriticMainModelParameters = nil
	
	NewAsynchronousAdvantageActorCriticModel.IsModelRunning = false
	
	return NewAsynchronousAdvantageActorCriticModel
	
end

function AsynchronousAdvantageActorCriticModel:setParameters(learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, rewardAveragingRate, totalNumberOfReinforcementsToUpdateMainModel)
	
	self.learningRate = learningRate or self.learningRate
	
	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.rewardAveragingRate = rewardAveragingRate or self.rewardAveragingRate
	
	self.totalNumberOfReinforcementsToUpdateMainModel = totalNumberOfReinforcementsToUpdateMainModel or self.totalNumberOfReinforcementsToUpdateMainModel
	
	for i = 1, #self.previousFeatureVectorArray, 1 do
		
		self.currentEpsilon[i] = epsilon or self.currentEpsilon[i]
		
	end
	
end

function AsynchronousAdvantageActorCriticModel:setClassesList(classesList)

	self.ClassesList = classesList

end

function AsynchronousAdvantageActorCriticModel:addActorCriticModel(ActorModel, CriticModel, ExperienceReplay)
	
	if not ActorModel then error("No actor model!") end
	
	if not CriticModel then error("No critic model!") end
	
	if self.ActorMainModelParameters then ActorModel:setModelParameters(self.ActorMainModelParameters) end
	
	if self.CriticMainModelParameters then CriticModel:setModelParameters(self.CriticMainModelParameters) end
	
	table.insert(self.ActorModelArray, ActorModel)
		
	table.insert(self.CriticModelArray, CriticModel)
	
	if ExperienceReplay then table.insert(self.ExperienceReplayArray, ExperienceReplay) end
	
	table.insert(self.episodeRewardArray,  0)

	table.insert(self.currentNumberOfReinforcementsArray,  0)

	table.insert(self.currentNumberOfEpisodesArray,  0)

	table.insert(self.currentEpsilonArray,  0)

	table.insert(self.runningRewardArray,  0)

	table.insert(self.advantageHistoryArray, {})

	table.insert(self.actionProbabilityHistoryArray, {})

	table.insert(self.criticValueHistoryArray, {})
	
end

local function calculateProbability(outputMatrix)

	local sumVector = AqwamMatrixLibrary:horizontalSum(outputMatrix)

	local result = AqwamMatrixLibrary:divide(outputMatrix, sumVector)

	return result

end

local function sampleAction(actionProbabilityVector)
	
	local totalProbability = 0
	
	for _, probability in ipairs(actionProbabilityVector[1]) do
		
		totalProbability += probability
		
	end

	local randomValue = math.random() * totalProbability

	local cumulativeProbability = 0
	
	local actionIndex = 1
	
	for i, probability in ipairs(actionProbabilityVector[1]) do
		
		cumulativeProbability += probability
		
		if (randomValue > cumulativeProbability) then continue end
			
		actionIndex = i
		
		break
		
	end
	
	return actionIndex
	
end

function AsynchronousAdvantageActorCriticModel:update(previousFeatureVector, action, rewardValue, currentFeatureVector, actorCriticModelNumber)
	
	local ActorModel = self.ActorModelArray[actorCriticModelNumber]
	
	local CriticModel = self.CriticModelArray[actorCriticModelNumber]
	
	if not ActorModel then error("No actor model!") end

	if not CriticModel then error("No critic model!") end

	local allOutputsMatrix = ActorModel:predict(previousFeatureVector, true)
	
	local actionProbabilityVector = calculateProbability(allOutputsMatrix)

	local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]
	
	local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]
	
	local advantageValue = rewardValue + (self.discountFactor * (currentCriticValue - currentCriticValue))
	
	local numberOfActions = #allOutputsMatrix[1]
	
	local actionIndex = sampleAction(actionProbabilityVector)
	
	local action = self.ClassesList[actionIndex]
	
	local actionProbability = actionProbabilityVector[1][actionIndex]
	
	self.episodeRewardArray[actorCriticModelNumber] += rewardValue
	
	table.insert(self.advantageHistoryArray[actorCriticModelNumber], advantageValue)
	
	table.insert(self.actionProbabilityHistoryArray[actorCriticModelNumber], actionProbability)
	
	table.insert(self.criticValueHistoryArray[actorCriticModelNumber], previousCriticValue)
	
	return allOutputsMatrix

end

function AsynchronousAdvantageActorCriticModel:episodeUpdate(numberOfFeatures, actorCriticModelNumber)

	self.runningRewardArray[actorCriticModelNumber] = (self.rewardAveragingRate * self.episodeRewardArray[actorCriticModelNumber]) + ((1 - self.rewardAveragingRate) * self.runningRewardArray[actorCriticModelNumber])
	
	local historyLength = #self.advantageHistoryArray[actorCriticModelNumber]
	
	local sumActorLosses = 0
	
	local sumCriticLosses = 0
	
	for h = 1, historyLength, 1 do
		
		local advantage = self.advantageHistoryArray[actorCriticModelNumber][h]
		
		local actionProbability = self.actionProbabilityHistoryArray[actorCriticModelNumber][h]
		
		local actorLoss = -math.log(actionProbability) * advantage
		
		local criticLoss = math.pow(advantage, 2)
		
		sumActorLosses += actorLoss
		
		sumCriticLosses += criticLoss
		
	end
	
	local lossValue = sumActorLosses + sumCriticLosses
	
	local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)
	local lossVector = AqwamMatrixLibrary:createMatrix(1, #self.ClassesList, lossValue)
	
	local ActorModel = self.ActorModelArray[actorCriticModelNumber]
	local CriticModel = self.CriticModelArray[actorCriticModelNumber]
	
	if not ActorModel then error("No actor model!") end

	if not CriticModel then error("No critic model!") end
	
	ActorModel:forwardPropagate(featureVector, true)
	CriticModel:forwardPropagate(featureVector, true)
	
	self.ActorModelCostFunctionDerivativesArray[actorCriticModelNumber] = ActorModel:backPropagate(sumActorLosses, true)
	self.CriticModelCostFunctionDerivativesArray[actorCriticModelNumber] = CriticModel:backPropagate(sumCriticLosses, true, true)
	
	------------------------------------------------------
	
	self.episodeRewardArray[actorCriticModelNumber] = 0

	self.currentNumberOfReinforcementsArray[actorCriticModelNumber] = 0

	self.currentNumberOfEpisodesArray[actorCriticModelNumber] += 1

	self.currentEpsilonArray[actorCriticModelNumber] *= self.epsilonDecayFactor
	
	table.clear(self.advantageHistoryArray[actorCriticModelNumber])
	
	table.clear(self.actionProbabilityHistoryArray[actorCriticModelNumber])
	
	table.clear(self.criticValueHistoryArray[actorCriticModelNumber])
	
end

function AsynchronousAdvantageActorCriticModel:fetchHighestValueInVector(outputVector)

	local highestValue, classIndex = AqwamMatrixLibrary:findMaximumValueInMatrix(outputVector)

	if (classIndex == nil) then return nil, highestValue end

	local predictedLabel = self.ClassesList[classIndex[2]]

	return predictedLabel, highestValue
	
end

function AsynchronousAdvantageActorCriticModel:getLabelFromOutputMatrix(outputMatrix)

	local predictedLabelVector = AqwamMatrixLibrary:createMatrix(#outputMatrix, 1)

	local highestValueVector = AqwamMatrixLibrary:createMatrix(#outputMatrix, 1)

	local highestValue

	local outputVector

	local classIndex

	local predictedLabel

	for i = 1, #outputMatrix, 1 do

		outputVector = {outputMatrix[i]}

		predictedLabel, highestValue = self:fetchHighestValueInVector(outputVector)

		predictedLabelVector[i][1] = predictedLabel

		highestValueVector[i][1] = highestValue

	end

	return predictedLabelVector, highestValueVector

end

function AsynchronousAdvantageActorCriticModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput, actorCriticModelNumber)
	
	actorCriticModelNumber = actorCriticModelNumber or Random.new():NextInteger(1, #self.currentEpsilonArray)

	if (self.currentNumberOfReinforcementsArray[actorCriticModelNumber] >= self.numberOfReinforcementsPerEpisode) then
		
		self:episodeUpdate(#currentFeatureVector[1], actorCriticModelNumber)

	end

	self.currentNumberOfReinforcementsArray[actorCriticModelNumber] += 1
	
	self.currentTotalNumberOfReinforcementsToUpdateMainModel += 1
	
	local action
	
	local actionIndex
	
	local actionVector

	local highestValue

	local highestValueVector

	local allOutputsMatrix = AqwamMatrixLibrary:createMatrix(1, #self.ClassesList)

	local randomProbability = Random.new():NextNumber()
	
	local previousFeatureVector = self.previousFeatureVectorArray[actorCriticModelNumber]
	
	local ExperienceReplay = self.ExperienceReplayArray[actorCriticModelNumber]

	if (randomProbability < self.currentEpsilonArray[actorCriticModelNumber]) then

		local randomNumber = Random.new():NextInteger(1, #self.ClassesList)

		action = self.ClassesList[randomNumber]

		allOutputsMatrix[1][randomNumber] = randomProbability

	else

		if (previousFeatureVector) then
			
			allOutputsMatrix = self:update(previousFeatureVector, action, rewardValue, currentFeatureVector, actorCriticModelNumber)
			
			actionVector, highestValueVector = self:getLabelFromOutputMatrix(allOutputsMatrix)

			action = actionVector[1][1]

			highestValue = highestValueVector[1][1]
			
		end

	end

	if (ExperienceReplay) and (previousFeatureVector) then 

		ExperienceReplay:addExperience(previousFeatureVector, action, rewardValue, currentFeatureVector)

		ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

			self:update(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector, actorCriticModelNumber)

		end)

	end

	self.previousFeatureVectorArray[actorCriticModelNumber] = currentFeatureVector

	if (returnOriginalOutput) then return allOutputsMatrix end

	return action, highestValue
	
end

function AsynchronousAdvantageActorCriticModel:setActorCriticMainModelParameters(ActorMainModelParameters, CriticMainModelParameters, applyToAllChildModels)
	
	self.ActorMainModelParameters = ActorMainModelParameters

	self.CriticMainModelParameters = CriticMainModelParameters
	
	if not applyToAllChildModels then return nil end
		
	for i = 1, #self.ActorModelArray, 1 do
			
		self.ActorModelArray[i]:setModelParameters(ActorMainModelParameters)
		
		self.CriticModelArray[i]:setModelParameters(CriticMainModelParameters)
			
	end
		
end

function AsynchronousAdvantageActorCriticModel:getActorCriticMainModelParameters()
	
	return self.ActorMainModelParameters, self.CriticMainModelParameters
	
end

function AsynchronousAdvantageActorCriticModel:start()
	
	if (self.IsModelRunning == true) then error("The model is already running!") end
	
	self.IsModelRunning = true
	
	local trainCoroutine = coroutine.create(function()

		repeat
			
			task.wait()
			
			if (self.currentTotalNumberOfReinforcementsToUpdateMainModel < self.totalNumberOfReinforcementsToUpdateMainModel) then continue end
			
			local ActorMainModelParameters = self.ActorMainModelParameters
			
			local CriticMainModelParameters = self.CriticMainModelParameters
			
			if not ActorMainModelParameters or not CriticMainModelParameters then
				
				local randomInteger = Random.new():NextInteger(1, #self.ActorModelArray)
				
				ActorMainModelParameters = self.ActorModelArray[randomInteger]:getModelParameters()
				CriticMainModelParameters = self.CriticModelArray[randomInteger]:getModelParameters()
				
			end
			
			if not ActorMainModelParameters or not CriticMainModelParameters then continue end

			self.currentTotalNumberOfReinforcementsToUpdateMainModel = 0
			
			local CriticMainModelCostFunctionDerivatives = {}
			
			for i = 1, #ActorMainModelParameters, 1 do
				
				local ActorMainModelCostFunctionDerivatives = AqwamMatrixLibrary:createMatrix(#ActorMainModelParameters[i], #ActorMainModelParameters[i][1])
				
				for _, ActorModelCostFunctionDerivatives in ipairs(self.ActorModelCostFunctionDerivativesArray) do

					ActorMainModelCostFunctionDerivatives = AqwamMatrixLibrary:add(ActorMainModelCostFunctionDerivatives, ActorModelCostFunctionDerivatives[i])

				end
				
				ActorMainModelCostFunctionDerivatives = AqwamMatrixLibrary:multiply(self.learningRate, ActorMainModelCostFunctionDerivatives)
				
				ActorMainModelParameters[i] = AqwamMatrixLibrary:subtract(ActorMainModelParameters[i], ActorMainModelCostFunctionDerivatives)
				
			end
			
			for i = 1, #CriticMainModelParameters, 1 do
				
				local CriticMainModelCostFunctionDerivatives = AqwamMatrixLibrary:createMatrix(#CriticMainModelParameters[i], #CriticMainModelParameters[i][1])
				
				for _, CriticModelCostFunctionDerivatives in ipairs(self.CriticModelCostFunctionDerivativesArray) do 

					CriticMainModelCostFunctionDerivatives = AqwamMatrixLibrary:add(CriticMainModelCostFunctionDerivatives, CriticModelCostFunctionDerivatives[i])

				end
				
				CriticMainModelCostFunctionDerivatives = AqwamMatrixLibrary:multiply(self.learningRate, CriticMainModelCostFunctionDerivatives)
				
				CriticMainModelParameters[i] = AqwamMatrixLibrary:subtract(CriticMainModelParameters[i], CriticMainModelCostFunctionDerivatives)

			end
			
			for _, ActorModel in ipairs(self.ActorModelArray) do ActorModel:setModelParameters(ActorMainModelParameters) end

			for _, CriticModel in ipairs(self.CriticModelArray) do CriticModel:setModelParameters(CriticMainModelParameters) end
			
			self.ActorMainModelParameters = ActorMainModelParameters

			self.CriticMainModelParameters = CriticMainModelParameters

		until (self.IsModelRunning == false)

	end)

	coroutine.resume(trainCoroutine)

	return trainCoroutine
	
end

function AsynchronousAdvantageActorCriticModel:stop()
	
	self.IsModelRunning = false
	
end

function AsynchronousAdvantageActorCriticModel:getCurrentNumberOfEpisodes(actorCriticModelNumber)

	return self.currentNumberOfEpisodesArray[actorCriticModelNumber]

end

function AsynchronousAdvantageActorCriticModel:getCurrentNumberOfReinforcements(actorCriticModelNumber)

	return self.currentNumberOfReinforcementsArray[actorCriticModelNumber]

end

function AsynchronousAdvantageActorCriticModel:getCurrentEpsilon(actorCriticModelNumber)

	return self.currentEpsilonArray[actorCriticModelNumber]

end

function AsynchronousAdvantageActorCriticModel:getCurrentTotalNumberOfReinforcementsToUpdateMainModel()

	return self.currentTotalNumberOfReinforcementsToUpdateMainModel

end

function AsynchronousAdvantageActorCriticModel:singleReset(actorCriticModelNumber)
	
	self.episodeRewardArray[actorCriticModelNumber] = 0
	
	self.runningRewardArray[actorCriticModelNumber] = 0

	self.currentNumberOfReinforcementsArray[actorCriticModelNumber] = 0

	self.currentNumberOfEpisodesArray[actorCriticModelNumber] = 0

	self.previousFeatureVectorArray[actorCriticModelNumber] = nil

	self.currentEpsilonArray[actorCriticModelNumber] = self.epsilon
	
	table.clear(self.advantageHistoryArray[actorCriticModelNumber])

	table.clear(self.actionProbabilityHistoryArray[actorCriticModelNumber])

	table.clear(self.criticValueHistoryArray[actorCriticModelNumber])
	
	local ExperienceReplay = self.ExperienceReplayArray[actorCriticModelNumber]

	if (ExperienceReplay) then ExperienceReplay:reset() end

end

function AsynchronousAdvantageActorCriticModel:reset()
	
	for i = 1, #self.currentEpsilonArray, 1 do self:singleReset(i) end
	
	self.currentTotalNumberOfReinforcementsToUpdateMainModel = 0
	
end

function AsynchronousAdvantageActorCriticModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return AsynchronousAdvantageActorCriticModel
