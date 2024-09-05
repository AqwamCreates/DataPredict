--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
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

local defaultTotalNumberOfReinforcementsToUpdateMainModel = 100

local defaultActionSelectionFunction = "Maximum"

function AsynchronousAdvantageActorCriticModel.new(learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, totalNumberOfReinforcementsToUpdateMainModel, actionSelectionFunction)

	local NewAsynchronousAdvantageActorCriticModel = {}

	setmetatable(NewAsynchronousAdvantageActorCriticModel, AsynchronousAdvantageActorCriticModel)

	NewAsynchronousAdvantageActorCriticModel.learningRate = learningRate or defaultLearningRate

	NewAsynchronousAdvantageActorCriticModel.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode

	NewAsynchronousAdvantageActorCriticModel.epsilon = epsilon or defaultEpsilon

	NewAsynchronousAdvantageActorCriticModel.epsilonDecayFactor =  epsilonDecayFactor or defaultEpsilonDecayFactor

	NewAsynchronousAdvantageActorCriticModel.discountFactor =  discountFactor or defaultDiscountFactor

	NewAsynchronousAdvantageActorCriticModel.currentEpsilonArray = {}

	NewAsynchronousAdvantageActorCriticModel.previousFeatureVectorArray = {}

	NewAsynchronousAdvantageActorCriticModel.printReinforcementOutput = true

	NewAsynchronousAdvantageActorCriticModel.currentNumberOfReinforcementsArray = {}

	NewAsynchronousAdvantageActorCriticModel.currentNumberOfEpisodesArray = {}

	NewAsynchronousAdvantageActorCriticModel.advantageValueHistoryArray = {}

	NewAsynchronousAdvantageActorCriticModel.actionProbabilityValueHistoryArray = {}

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

	NewAsynchronousAdvantageActorCriticModel.actionSelectionFunction = actionSelectionFunction or defaultActionSelectionFunction

	NewAsynchronousAdvantageActorCriticModel.ActorMainModelParameters = nil

	NewAsynchronousAdvantageActorCriticModel.CriticMainModelParameters = nil

	NewAsynchronousAdvantageActorCriticModel.IsModelRunning = false

	return NewAsynchronousAdvantageActorCriticModel

end

function AsynchronousAdvantageActorCriticModel:setParameters(learningRate, numberOfReinforcementsPerEpisode, epsilon, epsilonDecayFactor, discountFactor, totalNumberOfReinforcementsToUpdateMainModel, actionSelectionFunction)

	self.learningRate = learningRate or self.learningRate

	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode

	self.epsilon = epsilon or self.epsilon

	self.epsilonDecayFactor =  epsilonDecayFactor or self.epsilonDecayFactor

	self.discountFactor =  discountFactor or self.discountFactor

	self.totalNumberOfReinforcementsToUpdateMainModel = totalNumberOfReinforcementsToUpdateMainModel or self.totalNumberOfReinforcementsToUpdateMainModel

	self.actionSelectionFunction = actionSelectionFunction or self.actionSelectionFunction

	for i = 1, #self.previousFeatureVectorArray, 1 do

		self.currentEpsilon[i] = epsilon or self.currentEpsilon[i]

	end

end

function AsynchronousAdvantageActorCriticModel:setClassesList(ClassesList)

	self.ClassesList = ClassesList

end

function AsynchronousAdvantageActorCriticModel:addActorCriticModel(ActorModel, CriticModel, ExperienceReplay)

	if not ActorModel then error("No actor model!") end

	if not CriticModel then error("No critic model!") end

	ActorModel:setPrintOutput(false)

	CriticModel:setPrintOutput(false)

	if self.ActorMainModelParameters then ActorModel:setModelParameters(self.ActorMainModelParameters) end

	if self.CriticMainModelParameters then CriticModel:setModelParameters(self.CriticMainModelParameters) end

	table.insert(self.ActorModelArray, ActorModel)

	table.insert(self.CriticModelArray, CriticModel)

	if ExperienceReplay then table.insert(self.ExperienceReplayArray, ExperienceReplay) end

	table.insert(self.currentNumberOfReinforcementsArray,  0)

	table.insert(self.currentNumberOfEpisodesArray,  0)

	table.insert(self.currentEpsilonArray,  0)

	table.insert(self.advantageValueHistoryArray, {})

	table.insert(self.actionProbabilityValueHistoryArray, {})

	table.insert(self.criticValueHistoryArray, {})

end

local function calculateProbability(vector)

	local zScoreVector, standardDeviationVector = AqwamMatrixLibrary:horizontalZScoreNormalization(vector)

	local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)

	local probabilityVectorPart1 = AqwamMatrixLibrary:multiply(-0.5, squaredZScoreVector)

	local probabilityVectorPart2 = AqwamMatrixLibrary:exponent(probabilityVectorPart1)

	local probabilityVectorPart3 = AqwamMatrixLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))

	local probabilityVector = AqwamMatrixLibrary:divide(probabilityVectorPart2, probabilityVectorPart3)

	return probabilityVector

end

local function sample(vector)

	local totalProbability = 0

	for _, probability in ipairs(vector[1]) do

		totalProbability += probability

	end

	local randomValue = math.random() * totalProbability

	local cumulativeProbability = 0

	local index = 1

	for i, probability in ipairs(vector[1]) do

		cumulativeProbability += probability

		if (randomValue > cumulativeProbability) then continue end

		index = i

		break

	end

	return index

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

	local advantageValue = rewardValue + (self.discountFactor * currentCriticValue) - previousCriticValue

	local numberOfActions = #allOutputsMatrix[1]

	local actionIndex = table.find(ActorModel:getClassesList(), action)

	local actionProbabilityValue = actionProbabilityVector[1][actionIndex]
	
	local logActionProbabilityValue = math.log(actionProbabilityValue)

	table.insert(self.advantageValueHistoryArray[actorCriticModelNumber], advantageValue)

	table.insert(self.actionProbabilityValueHistoryArray[actorCriticModelNumber], logActionProbabilityValue)

	table.insert(self.criticValueHistoryArray[actorCriticModelNumber], previousCriticValue)

	return allOutputsMatrix

end

function AsynchronousAdvantageActorCriticModel:episodeUpdate(actorCriticModelNumber)
	
	local advantageValueHistory = self.advantageValueHistoryArray[actorCriticModelNumber]
	
	local actionProbabilityValueHistory =  self.actionProbabilityValueHistoryArray[actorCriticModelNumber]

	local historyLength = #advantageValueHistory

	local sumActorLoss = 0

	local sumCriticLoss = 0

	for h = 1, historyLength, 1 do

		local advantage = advantageValueHistory[h]

		local actionProbabilityValue = actionProbabilityValueHistory[h]

		local actorLoss = actionProbabilityValue * advantage

		sumActorLoss = sumActorLoss + actorLoss

		sumCriticLoss = sumCriticLoss + advantage

	end

	local ActorModel = self.ActorModelArray[actorCriticModelNumber]
	
	local CriticModel = self.CriticModelArray[actorCriticModelNumber]

	if not ActorModel then error("No actor model!") end
	
	if not CriticModel then error("No critic model!") end

	local numberOfFeatures, hasBias = ActorModel:getLayer(1)

	numberOfFeatures += (hasBias and 1) or 0

	local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

	ActorModel:forwardPropagate(featureVector, true)
	
	CriticModel:forwardPropagate(featureVector, true)

	self.ActorModelCostFunctionDerivativesArray[actorCriticModelNumber] = ActorModel:calculateCostFunctionDerivativeMatrixTable(-sumActorLoss, true)
	
	self.CriticModelCostFunctionDerivativesArray[actorCriticModelNumber] = CriticModel:calculateCostFunctionDerivativeMatrixTable(-sumCriticLoss, true)

	------------------------------------------------------

	self.currentNumberOfReinforcementsArray[actorCriticModelNumber] = 0

	self.currentNumberOfEpisodesArray[actorCriticModelNumber] = self.currentNumberOfEpisodesArray[actorCriticModelNumber] + 1

	self.currentEpsilonArray[actorCriticModelNumber] = self.currentEpsilonArray[actorCriticModelNumber] * self.epsilonDecayFactor

	table.clear(self.advantageValueHistoryArray[actorCriticModelNumber])

	table.clear(self.actionProbabilityValueHistoryArray[actorCriticModelNumber])

	table.clear(self.criticValueHistoryArray[actorCriticModelNumber])

end

function AsynchronousAdvantageActorCriticModel:fetchHighestValueInVector(outputVector)

	local highestValue, classIndex = AqwamMatrixLibrary:findMaximumValue(outputVector)

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

function AsynchronousAdvantageActorCriticModel:predict(currentFeatureVector, returnOriginalOutput, actorCriticModelNumber)

	local Model = self.ActorModelArray[actorCriticModelNumber]

	return Model:predict(currentFeatureVector, returnOriginalOutput)

end

function AsynchronousAdvantageActorCriticModel:selectAction(currentFeatureVector, classesList, actorCriticModelNumber)

	local actionVector = self:predict(currentFeatureVector, true, actorCriticModelNumber)

	local actionSelectionFunction = self.actionSelectionFunction

	local action

	local selectedValue

	if (actionSelectionFunction == "Maximum") then

		local actionVector, selectedValueVector = self:getLabelFromOutputMatrix(actionVector)

		action = actionVector[1][1]

		selectedValue = selectedValueVector[1][1]

	elseif (actionSelectionFunction == "Sample") then

		local actionProbabilityVector = calculateProbability(actionVector)

		local actionIndex = sample(actionProbabilityVector)

		action = classesList[actionIndex]

		selectedValue = actionVector[1][actionIndex]

	end

	return action, selectedValue, actionVector

end

function AsynchronousAdvantageActorCriticModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput, actorCriticModelNumber)

	actorCriticModelNumber = actorCriticModelNumber or Random.new():NextInteger(1, #self.currentEpsilonArray)

	self.currentNumberOfReinforcementsArray[actorCriticModelNumber] += 1

	self.currentTotalNumberOfReinforcementsToUpdateMainModel += 1
	
	local previousFeatureVector = self.previousFeatureVectorArray[actorCriticModelNumber]

	local ExperienceReplay = self.ExperienceReplayArray[actorCriticModelNumber]

	local currrentEpsilon = self.currentEpsilonArray[actorCriticModelNumber]

	local ClassesList = self.ClassesList

	local action

	local actionIndex

	local actionVector

	local selectedValue
	
	local temporalDifferenceError

	local actionVector = AqwamMatrixLibrary:createMatrix(1, #ClassesList)

	local randomProbability = Random.new():NextNumber()

	if (randomProbability < currrentEpsilon) then

		local numberOfClasses = #ClassesList

		local randomNumber = Random.new():NextInteger(1, numberOfClasses)

		action = ClassesList[randomNumber]

		actionVector = AqwamMatrixLibrary:createMatrix(1, numberOfClasses)

		actionVector[1][randomNumber] = randomProbability

	else

		action, selectedValue, actionVector = self:selectAction(currentFeatureVector, ClassesList, actorCriticModelNumber)

	end

	if (previousFeatureVector) then 

		temporalDifferenceError = self:update(previousFeatureVector, action, rewardValue, currentFeatureVector, actorCriticModelNumber) 

	end

	if (self.currentNumberOfReinforcementsArray[actorCriticModelNumber] >= self.numberOfReinforcementsPerEpisode) then

		self:episodeUpdate(#currentFeatureVector[1], actorCriticModelNumber)

	end

	if (ExperienceReplay) and (previousFeatureVector) then 

		ExperienceReplay:addExperience(previousFeatureVector, action, rewardValue, currentFeatureVector)

		ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

		ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

			return self:update(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector, actorCriticModelNumber)

		end)

	end

	self.previousFeatureVectorArray[actorCriticModelNumber] = currentFeatureVector

	if (returnOriginalOutput) then return actionVector end

	return action, selectedValue

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

function AsynchronousAdvantageActorCriticModel:getActorCriticModels(actorCriticModelNumber)

	return self.ActorModelArray[actorCriticModelNumber], self.CriticModelArray[actorCriticModelNumber]

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

function AsynchronousAdvantageActorCriticModel:reset(actorCriticModelNumber)

	self.currentNumberOfReinforcementsArray[actorCriticModelNumber] = 0

	self.currentNumberOfEpisodesArray[actorCriticModelNumber] = 0

	self.previousFeatureVectorArray[actorCriticModelNumber] = nil

	self.currentEpsilonArray[actorCriticModelNumber] = self.epsilon

	table.clear(self.advantageValueHistoryArray[actorCriticModelNumber])

	table.clear(self.actionProbabilityValueHistoryArray[actorCriticModelNumber])

	table.clear(self.criticValueHistoryArray[actorCriticModelNumber])

	local ExperienceReplay = self.ExperienceReplayArray[actorCriticModelNumber]

	if (ExperienceReplay) then ExperienceReplay:reset() end

end

function AsynchronousAdvantageActorCriticModel:resetAll()

	for i = 1, #self.currentEpsilonArray, 1 do self:reset(i) end

	self.currentTotalNumberOfReinforcementsToUpdateMainModel = 0

end

function AsynchronousAdvantageActorCriticModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return AsynchronousAdvantageActorCriticModel