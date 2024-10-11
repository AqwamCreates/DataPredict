local ServerScriptService = game:GetService("ServerScriptService")

local DataPredict = require(ServerScriptService.AqwamMachineAndDeepLearningLibrary)

local MatrixL =  require(ServerScriptService.AqwamMatrixLibrary)

local ReinforcementLearningQuickSetup = DataPredict.QuickSetups.CategoricalPolicy.new(10, 1, "Sample")

ReinforcementLearningQuickSetup:setEpsilonValueScheduler(DataPredict.ValueSchedulers.TimeDecay.new(0.85))

ReinforcementLearningQuickSetup:setPrintOutput(false)

local maxRewardArrayLength = 100

local maxCurrentArrayLength = 100

local isRewardedArray = {}

local currentAccuracyArray = {}

local classesList = {1, 2, 3, 4}

local ExperienceReplay = DataPredict.ExperienceReplays.PrioritizedExperienceReplay.new(10, 15, 20)



ReinforcementLearningQuickSetup:setClassesList(classesList)

local function buildDiscriminator()
	
	local NN = DataPredict.Models.NeuralNetwork.new()
	
	NN:addLayer(4, true, "None", 0.00001)
	
	NN:addLayer(7, true, "Tanh", 0.00001)

	NN:addLayer(1, false, "Sigmoid")
	
	return NN
	
end

local function buildModel()
	
	--local ExperienceReplay = DataPredict.ExperienceReplays.PrioritizedExperienceReplay.new(5, 10)

	local Model = DataPredict.Models.NeuralNetwork.new(1)
	
	ExperienceReplay:setModel(Model)

	Model:addLayer(2, true, "ELU", 0.0001)
	
	Model:addLayer(7, true, "ELU", 0.0001)

	Model:addLayer(4, false, "StableSoftmax", 0.0001)
	
	Model:setClassesList(classesList)
	
	local RLModel = DataPredict.Models.DeepDoubleQLearningV1.new()
	
	RLModel:setModel(Model)
	
	return RLModel

end

local function buildModel2()

	local ACModel = DataPredict.Models.ProximalPolicyOptimization.new()
	
	--ACModel:setExperienceReplay(DataPredict.ExperienceReplays.UniformExperienceReplay.new())

	--ACModel:setExperienceReplay(DataPredict.ExperienceReplays.UniformExperienceReplay.new(3, 10))

	--ACModel:setPrintReinforcementOutput(false)

	local AModel = DataPredict.Models.NeuralNetwork.new(1)

	AModel:setPrintOutput(false)

	AModel:addLayer(2, true)

	AModel:addLayer(4, false, "Tanh", 0.0001)
	
	AModel:setClassesList(classesList)

	local CModel = DataPredict.Models.NeuralNetwork.new(1)

	CModel:setPrintOutput(false)

	CModel:addLayer(2, true)

	CModel:addLayer(1, false, "Tanh", 0.0001)

	ACModel:setActorModel(AModel)
	
	ACModel:setCriticModel(CModel)
	
	--ACModel:setClassesList(classesList)

	return ACModel

end

local function buildModel3()
	
	local NeuralNetwork = DataPredict.Models.NeuralNetwork
	
	local ACModel = DataPredict.Models.AsynchronousAdvantageActorCritic.new(0.01, nil, 0, 1, 1)
	
	--ACModel:setExperienceReplay(DataPredict.ExperienceReplays.UniformExperienceReplay.new(3, 10))
	
	ACModel:setClassesList({1, 2, 3, 4})
	
	--ACModel:setPrintReinforcementOutput(false)
	
	local AModel = NeuralNetwork.new(1, 0.01)
	
	AModel:setClassesList({1, 2, 3, 4})
	
	AModel:addLayer(2, true, "Tanh")
	
	AModel:addLayer(4, false, "StableSoftmax")
	
	local CModel = NeuralNetwork.new(1, 0.01)

	CModel:addLayer(2, true, "Tanh")

	CModel:addLayer(1, false, "None")
	
	ACModel:addActorCriticModel(AModel, CModel) --, DataPredict.ExperienceReplays.UniformExperienceReplay.new(3, 10))
	
	local AModel2 = NeuralNetwork.new(1, 0.01)

	AModel2:setClassesList({1, 2, 3, 4})

	AModel2:addLayer(2, true, "Tanh")

	AModel2:addLayer(4, false, "StableSoftmax")

	local CModel2 = NeuralNetwork.new(1, 0.01)

	CModel2:addLayer(2, true, "Tanh")

	CModel2:addLayer(1, false, "None")

	ACModel:addActorCriticModel(AModel2, CModel2) --, DataPredict.ExperienceReplays.UniformExperienceReplay.new(3, 10))
	
	ACModel:start()

	return ACModel
	
end

local function buildModel4()

	local DuelingQLearningModel = DataPredict.Models.DeepDoubleDuelingQLearningV2.new()

	--ACModel:setExperienceReplay(DataPredict.ExperienceReplays.UniformExperienceReplay.new(3, 10))

	--ACModel:setPrintReinforcementOutput(false)
	
	local SModel = DataPredict.Models.NeuralNetwork.new(1, 0.01)
	
	SModel:addLayer(2, true, "ELU")

	SModel:addLayer(3, true, "ELU")

	SModel:addLayer(4, false, "ELU")

	local AModel = DataPredict.Models.NeuralNetwork.new(1, 0.01)

	AModel:setClassesList({1, 2, 3, 4})

	AModel:setPrintOutput(false)

	AModel:addLayer(4, true, "ELU")
	
	AModel:addLayer(3, true, "ELU")

	AModel:addLayer(4, false, "StableSoftmax")

	local ValueModel = DataPredict.Models.NeuralNetwork.new(1, 0.01)

	ValueModel:setPrintOutput(false)

	ValueModel:addLayer(4, true, "ELU")
	
	AModel:addLayer(3, true, "ELU")

	ValueModel:addLayer(1, false, "None")
	
	ValueModel:generateLayers()
	
	DuelingQLearningModel:setSharedModel(SModel)

	DuelingQLearningModel:setAdvantageModel(AModel)

	DuelingQLearningModel:setValueModel(ValueModel)
	
	task.wait(1)

	return DuelingQLearningModel

end

local function buildModel5()

	local ACModel = DataPredict.AqwamCustomModels.DeepConfidenceQLearning.new()

	--ACModel:setExperienceReplay(DataPredict.ExperienceReplays.UniformExperienceReplay.new())

	--ACModel:setExperienceReplay(DataPredict.ExperienceReplays.UniformExperienceReplay.new(3, 10))

	--ACModel:setPrintReinforcementOutput(false)

	local AModel = DataPredict.Models.NeuralNetwork.new(1)

	AModel:setPrintOutput(false)

	AModel:addLayer(2, true)

	AModel:addLayer(4, false, "Tanh", 0.0001)

	AModel:setClassesList(classesList)

	local CModel = DataPredict.Models.NeuralNetwork.new(1)

	CModel:setPrintOutput(false)

	CModel:addLayer(4, false)

	CModel:addLayer(1, false, "Tanh", 0.1)

	ACModel:setActorModel(AModel)

	ACModel:setConfidenceModel(CModel)

	--ACModel:setClassesList(classesList)

	return ACModel

end

local function createExpertActionMatrix(environmentFeatureMatrix, predictedLabel)
	
	local numberOfData = #environmentFeatureMatrix
	
	local expertActionMatrix = MatrixL:createMatrix(numberOfData, 4, 0)
	
	for i = 1, numberOfData, 1 do
		
		local firstValue = environmentFeatureMatrix[i][2]
		
		local secondValue = environmentFeatureMatrix[i][3]
		
		if (firstValue >= 0) and (secondValue >= 0) then --  positive + positive = 1

			expertActionMatrix[i][1] = 1

		elseif (firstValue >= 0) and (secondValue < 0) then --  positive + negative = 2

			expertActionMatrix[i][2] = 1

		elseif (firstValue < 0) and (secondValue >= 0) then --  negative + positive = 3

			expertActionMatrix[i][3] = 1

		elseif (firstValue < 0) and (secondValue < 0) then --  negative + negative = 4

			expertActionMatrix[i][4] = 1

		end
		
	end

	return expertActionMatrix

end

local function generateEnvironmentFeatureMatrix(numberOfData)

	local featureMatrix1 = MatrixL:createRandomUniformMatrix(numberOfData, 3)

	local featureMatrix2 = MatrixL:createRandomUniformMatrix(numberOfData, 3)

	local environmentFeatureMatrix = MatrixL:subtract(featureMatrix1, featureMatrix2)
	
	for i = 1, numberOfData, 1 do
		
		environmentFeatureMatrix[i][1] = 1 -- 1 at first column for bias.
		
	end

	return environmentFeatureMatrix

end

local function countTrueBooleansInBooleanArray(booleanArray)

	local numberOfTrueBooleans = 0

	for i, boolean in ipairs(booleanArray) do

		if (boolean == true) then

			numberOfTrueBooleans += 1

		end

	end

	return numberOfTrueBooleans

end

local function removeRewards()

	local currentAccuracy

	local currentAverageAccuracy

	if (#isRewardedArray > maxRewardArrayLength) then

		table.remove(isRewardedArray, 1)

	end

end

local function getTotalReward()
	
	local totalReward = 0
	
	for _, reward in isRewardedArray do
		
		totalReward += reward
		
	end
	
	return totalReward
	
end

local function startEnvironment()

	local reward = 0
	local defaultReward = 1
	local defaultPunishment = -0.01
	local predictedLabel
	local isRewarded
	local totalReward = 0 -- Initialize total reward for the episode
	local steps = 0 -- Initialize steps counter for the episode
	local maxSteps = 100
	
	local discriminator = buildDiscriminator()
	
	local model = buildModel()
	
	local GAIL = DataPredict.Others.GenerativeAdversarialImitationLearningNetwork.new(1)
	
	GAIL:setDiscriminatorModel(discriminator)
	
	GAIL:setReinforcementLearningModel(model)
	
	GAIL:setClassesList({1, 2, 3, 4})
	
	local previousEnvironmentMatrix = generateEnvironmentFeatureMatrix(10)

	local environmentFeatureMatrix

	while true do
		
		environmentFeatureMatrix = generateEnvironmentFeatureMatrix(10)
		
		local expertActionMatrix = createExpertActionMatrix(environmentFeatureMatrix)
		
		GAIL:categoricalTrain(previousEnvironmentMatrix, expertActionMatrix, environmentFeatureMatrix)
		
		previousEnvironmentMatrix = environmentFeatureMatrix

	end

end

startEnvironment()
