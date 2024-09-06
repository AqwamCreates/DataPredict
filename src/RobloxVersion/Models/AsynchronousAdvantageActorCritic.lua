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

local DataPredict = script.Parent.Parent

local AqwamMatrixLibrary = require(DataPredict.AqwamMatrixLibraryLinker.Value)

AsynchronousAdvantageActorCriticModel = {}

AsynchronousAdvantageActorCriticModel.__index = AsynchronousAdvantageActorCriticModel

local defaultLearningRate = 0.1

local defaultNumberOfReinforcementsPerEpisode = 10

local defaultEpsilon = 0.5

local defaultEpsilonDecayFactor = 0.999

local defaultDiscountFactor = 0.95

local defaultTotalNumberOfReinforcementsToUpdateMainModel = 100

local defaultActionSelectionFunction = "Maximum"

local defaultPolicyMode = "Categorical"

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

	NewAsynchronousAdvantageActorCriticModel.actionProbabilityVectorHistoryArray = {}

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

	if self.ActorMainModelParameters then ActorModel:setModelParameters(self.ActorMainModelParameters) end

	if self.CriticMainModelParameters then CriticModel:setModelParameters(self.CriticMainModelParameters) end

	table.insert(self.ActorModelArray, ActorModel)

	table.insert(self.CriticModelArray, CriticModel)

	if ExperienceReplay then table.insert(self.ExperienceReplayArray, ExperienceReplay) end

	table.insert(self.currentNumberOfReinforcementsArray, 0)

	table.insert(self.currentNumberOfEpisodesArray, 0)

	table.insert(self.currentEpsilonArray, 0)

	table.insert(self.advantageValueHistoryArray, {})

	table.insert(self.actionProbabilityVectorHistoryArray, {})

end

local function selectIndexWithHighestValue(vector)

	local selectedIndex = 1

	local highestValue = -math.huge

	for index, value in ipairs(vector[1]) do

		if (highestValue > value) then

			highestValue = value

			selectedIndex = index

		end

	end

	return selectedIndex

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

	local probabilityVector = calculateProbability(vector)

	local totalProbability = 0

	for _, probability in ipairs(probabilityVector[1]) do

		totalProbability = totalProbability + probability

	end

	local randomValue = math.random() * totalProbability

	local cumulativeProbability = 0

	local selectedIndex = 1

	for i, probability in ipairs(probabilityVector[1]) do

		cumulativeProbability += probability

		if (randomValue > cumulativeProbability) then continue end

		selectedIndex = i

		break

	end

	return selectedIndex

end

function AsynchronousAdvantageActorCriticModel:categoricalUpdate(previousFeatureVector, action, rewardValue, currentFeatureVector, actorCriticModelNumber)

	local ActorModel = self.ActorModelArray[actorCriticModelNumber]

	local CriticModel = self.CriticModelArray[actorCriticModelNumber]

	if not ActorModel then error("No actor model!") end

	if not CriticModel then error("No critic model!") end

	local allOutputsMatrix = ActorModel:predict(previousFeatureVector, true)

	local actionProbabilityVector = calculateProbability(allOutputsMatrix)

	local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

	local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

	local advantageValue = rewardValue + (self.discountFactor * currentCriticValue) - previousCriticValue

	local logActionProbabilityVector = AqwamMatrixLibrary:logarithm(actionProbabilityVector)
	
	table.insert(self.actionProbabilityVectorHistoryArray[actorCriticModelNumber], logActionProbabilityVector)

	table.insert(self.advantageValueHistoryArray[actorCriticModelNumber], advantageValue)

	return advantageValue

end

function AsynchronousAdvantageActorCriticModel:diagonalGaussianUpdate(previousFeatureVector, expectedActionVector, rewardValue, currentFeatureVector, actorCriticModelNumber, standardDeviationVector)

	local ActorModel = self.ActorModelArray[actorCriticModelNumber]

	local CriticModel = self.CriticModelArray[actorCriticModelNumber]

	if not ActorModel then error("No actor model!") end

	if not CriticModel then error("No critic model!") end

	local randomNormalVector = AqwamMatrixLibrary:createRandomNormalMatrix(1, #expectedActionVector[1])

	local actionVectorPart1 = AqwamMatrixLibrary:multiply(standardDeviationVector, randomNormalVector)

	local actionVector = AqwamMatrixLibrary:add(expectedActionVector, actionVectorPart1)

	local zScoreVectorPart1 = AqwamMatrixLibrary:subtract(actionVector, expectedActionVector)

	local zScoreVector = AqwamMatrixLibrary:divide(zScoreVectorPart1, standardDeviationVector)

	local squaredZScoreVector = AqwamMatrixLibrary:power(zScoreVector, 2)

	local logActionProbabilityVectorPart1 = AqwamMatrixLibrary:logarithm(standardDeviationVector)

	local logActionProbabilityVectorPart2 = AqwamMatrixLibrary:multiply(2, logActionProbabilityVectorPart1)

	local logActionProbabilityVectorPart3 = AqwamMatrixLibrary:add(squaredZScoreVector, logActionProbabilityVectorPart2)

	local logActionProbabilityVector = AqwamMatrixLibrary:add(logActionProbabilityVectorPart3, math.log(2 * math.pi))

	local previousCriticValue = CriticModel:predict(previousFeatureVector, true)[1][1]

	local currentCriticValue = CriticModel:predict(currentFeatureVector, true)[1][1]

	local advantageValue = rewardValue + (self.discountFactor * currentCriticValue) - previousCriticValue
	
	table.insert(self.actionProbabilityVectorHistoryArray[actorCriticModelNumber], logActionProbabilityVector)

	table.insert(self.advantageValueHistoryArray[actorCriticModelNumber], advantageValue)

	return advantageValue

end

function AsynchronousAdvantageActorCriticModel:episodeUpdate(actorCriticModelNumber)

	local advantageValueHistory = self.advantageValueHistoryArray[actorCriticModelNumber]

	local actionProbabilityVectorHistory =  self.actionProbabilityVectorHistoryArray[actorCriticModelNumber]

	local historyLength = #advantageValueHistory

	local sumActorLossVector = AqwamMatrixLibrary:createMatrix(#actionProbabilityVectorHistory[1], #actionProbabilityVectorHistory[1][1], 0)

	local sumCriticLoss = 0

	for h = 1, historyLength, 1 do

		local advantageValue = advantageValueHistory[h]

		local actorLossVector = AqwamMatrixLibrary:multiply(actionProbabilityVectorHistory[h], advantageValue)

		sumCriticLoss = sumCriticLoss + advantageValue

		sumActorLossVector = AqwamMatrixLibrary:add(sumActorLossVector, actorLossVector)

	end
	
	sumActorLossVector = AqwamMatrixLibrary:unaryMinus(sumActorLossVector)

	local ActorModel = self.ActorModelArray[actorCriticModelNumber]

	local CriticModel = self.CriticModelArray[actorCriticModelNumber]

	if not ActorModel then error("No actor model!") end

	if not CriticModel then error("No critic model!") end

	local numberOfFeatures = ActorModel:getTotalNumberOfNeurons(1)

	local featureVector = AqwamMatrixLibrary:createMatrix(1, numberOfFeatures, 1)

	ActorModel:forwardPropagate(featureVector, true)

	CriticModel:forwardPropagate(featureVector, true)

	self.ActorModelCostFunctionDerivativesArray[actorCriticModelNumber] = ActorModel:calculateCostFunctionDerivativeMatrixTable(sumActorLossVector, true)

	self.CriticModelCostFunctionDerivativesArray[actorCriticModelNumber] = CriticModel:calculateCostFunctionDerivativeMatrixTable(-sumCriticLoss, true)

	------------------------------------------------------

	self.currentNumberOfReinforcementsArray[actorCriticModelNumber] = 0

	self.currentNumberOfEpisodesArray[actorCriticModelNumber] = self.currentNumberOfEpisodesArray[actorCriticModelNumber] + 1

	self.currentEpsilonArray[actorCriticModelNumber] = self.currentEpsilonArray[actorCriticModelNumber] * self.epsilonDecayFactor
	
	table.clear(actionProbabilityVectorHistory)

	table.clear(advantageValueHistory)

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

local selectActionFunctionList = {

	["Maximum"] = selectIndexWithHighestValue,

	["Sample"] = sample,

}

function AsynchronousAdvantageActorCriticModel:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput, actorCriticModelNumber, policyMode, standardDeviationVector)

	actorCriticModelNumber = actorCriticModelNumber or Random.new():NextInteger(1, #self.currentEpsilonArray)

	self.currentNumberOfReinforcementsArray[actorCriticModelNumber] = self.currentNumberOfReinforcementsArray[actorCriticModelNumber] + 1

	self.currentTotalNumberOfReinforcementsToUpdateMainModel = self.currentTotalNumberOfReinforcementsToUpdateMainModel + 1

	local actionSelectionFunction = self.actionSelectionFunction

	local previousFeatureVector = self.previousFeatureVectorArray[actorCriticModelNumber]

	local ExperienceReplay = self.ExperienceReplayArray[actorCriticModelNumber]

	local currentEpsilon = self.currentEpsilonArray[actorCriticModelNumber]

	local ActorModel = self.ActorModelArray[actorCriticModelNumber]

	local ClassesList = self.ClassesList

	local action

	local actionIndex

	local actionVector

	local actionValue

	local temporalDifferenceError

	local actionVector = ActorModel:predict(currentFeatureVector, true, actorCriticModelNumber)

	local randomProbability = Random.new():NextNumber()

	policyMode = policyMode or defaultPolicyMode

	if (policyMode == "Categorical") and (previousFeatureVector) then

		if (randomProbability < currentEpsilon) then

			actionIndex = Random.new():NextInteger(1, #ClassesList)

		else

			actionIndex = selectActionFunctionList[actionSelectionFunction](actionVector)

		end

		action = ClassesList[actionIndex]

		actionValue = actionVector[1][actionIndex]

		temporalDifferenceError = self:categoricalUpdate(previousFeatureVector, action, rewardValue, currentFeatureVector, actorCriticModelNumber) 

	elseif (policyMode == "DiagonalGaussian") and (previousFeatureVector) then 

		temporalDifferenceError = self:diagonalGaussianUpdate(previousFeatureVector, actionVector, rewardValue, currentFeatureVector, actorCriticModelNumber, standardDeviationVector) 

	end

	if (self.currentNumberOfReinforcementsArray[actorCriticModelNumber] >= self.numberOfReinforcementsPerEpisode) then

		self:episodeUpdate(actorCriticModelNumber)

	end

	if (policyMode == "Categorical") and (ExperienceReplay) and (previousFeatureVector) then 

		ExperienceReplay:addExperience(previousFeatureVector, action, rewardValue, currentFeatureVector)

		ExperienceReplay:addTemporalDifferenceError(temporalDifferenceError)

		ExperienceReplay:run(function(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector)

			return self:update(storedPreviousFeatureVector, storedAction, storedRewardValue, storedCurrentFeatureVector, actorCriticModelNumber)

		end)

	end

	self.previousFeatureVectorArray[actorCriticModelNumber] = currentFeatureVector

	if (returnOriginalOutput) then return actionVector end

	return action, actionValue

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
	
	table.clear(self.actionProbabilityVectorHistoryArray[actorCriticModelNumber])

	table.clear(self.advantageValueHistoryArray[actorCriticModelNumber])

	local ExperienceReplay = self.ExperienceReplayArray[actorCriticModelNumber]

	if (ExperienceReplay) then ExperienceReplay:reset() end

end

function AsynchronousAdvantageActorCriticModel:resetAll()

	for i, _ in ipairs(self.ActorModelArray) do self:reset(i) end

	self.currentTotalNumberOfReinforcementsToUpdateMainModel = 0

end

function AsynchronousAdvantageActorCriticModel:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return AsynchronousAdvantageActorCriticModel