local ServerScriptService = game:GetService("ServerScriptService")

local DataPredict = require(ServerScriptService.AqwamRobloxMachineAndDeepLearningLibrary)

local MatrixL =  require(ServerScriptService.MatrixL)

local maxRewardArrayLength = 100

local maxCurrentArrayLength = 100

local isRewardedArray = {}

local currentAccuracyArray = {}

local function buildModel()

	local Model = DataPredict.Models.QLearningNeuralNetwork.new(1, 0.1, nil, 100)
	
	Model:setExperienceReplay(DataPredict.ExperienceReplays.UniformExperienceReplay.new(3, 10))

	Model:addLayer(2, true, "Tanh")

	Model:addLayer(4, false, "None")

	Model:setClassesList({1, 2, 3, 4})

	Model:setPrintReinforcementOutput(false)

	return Model

end

local function buildModel2()
	
	local ACModel = DataPredict.Models.AdvantageActorCritic.new()
	
	ACModel:setClassesList({1, 2, 3, 4})
	
	ACModel:setPrintReinforcementOutput(false)
	
	local AModel = DataPredict.Models.NeuralNetwork.new(1, 0.1)
	
	AModel:setClassesList({1, 2, 3, 4})
	
	AModel:setPrintOutput(false)
	
	AModel:addLayer(2, true, "LeakyReLU")
	
	AModel:addLayer(4, false, "Softmax")
	
	local CModel = DataPredict.Models.NeuralNetwork.new(1, 0.1)

	CModel:setPrintOutput(false)

	CModel:addLayer(2, true, "LeakyReLU")

	CModel:addLayer(1, false, "None")
	
	ACModel:setActorModel(AModel)
	
	ACModel:setCriticModel(CModel)
	
	return ACModel
	
end

local function checkIfPunishedOrRewarded(environmentFeatureVector, predictedLabel)

	local isRewarded = nil

	if (environmentFeatureVector[1][2] >= 0) and (environmentFeatureVector[1][3] >= 0) then --  positive + positive = 1

		isRewarded = (predictedLabel == 1)

	elseif (environmentFeatureVector[1][2] >= 0) and (environmentFeatureVector[1][3] < 0) then --  positive + negative = 2

		isRewarded = (predictedLabel == 2)

	elseif (environmentFeatureVector[1][2] < 0) and (environmentFeatureVector[1][3] >= 0) then --  negative + positive = 3

		isRewarded = (predictedLabel == 3)

	elseif (environmentFeatureVector[1][2] < 0) and (environmentFeatureVector[1][3] < 0) then --  negative + negative = 4

		isRewarded = (predictedLabel == 4)

	end

	return isRewarded

end

local function generateEnvironmentFeatureVector()

	local featureVector1 = MatrixL:createRandomNormalMatrix(1, 3)

	local featureVector2 = MatrixL:createRandomNormalMatrix(1, 3)

	local environmentFeatureVector = MatrixL:subtract(featureVector1, featureVector2)

	environmentFeatureVector[1][1] = 1 -- 1 at first column for bias.

	return environmentFeatureVector

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

local function calculateCurrentAccuracy(booleanArray)

	local numberOfBooleans = #booleanArray

	local numberOfTrueBooleans = countTrueBooleansInBooleanArray(booleanArray) 

	local currentAccuracy = (numberOfTrueBooleans / numberOfBooleans) * 100

	currentAccuracy = math.floor(currentAccuracy)

	return currentAccuracy

end

local function calculateAverage(array)

	local sum = 0

	local average

	for i, number in ipairs(array) do

		sum += number

	end

	average = (sum / #array)

	return average

end

local function getCurrentAverageAccuracy()

	local currentAccuracy

	local currentAverageAccuracy

	if (#isRewardedArray > maxRewardArrayLength) then

		table.remove(isRewardedArray, 1)

		currentAccuracy = calculateCurrentAccuracy(isRewardedArray)

		table.insert(currentAccuracyArray, currentAccuracy)

	end

	if (#currentAccuracyArray > maxCurrentArrayLength) then

		table.remove(currentAccuracyArray, 1)

		currentAverageAccuracy = calculateAverage(currentAccuracyArray)

	end

	return currentAverageAccuracy

end

local function startEnvironment(Model)

	local reward = 0

	local defaultReward = 1

	local defaultPunishment = -0.01

	local predictedLabel

	local environmentFeatureVector

	local isRewarded

	local currentAverageAccuracy

	while true do

		environmentFeatureVector = generateEnvironmentFeatureVector()
		
		local initialTime = os.clock()

		predictedLabel = Model:reinforce(environmentFeatureVector, reward)
		
		local timeTaken = os.clock() - initialTime

		isRewarded = checkIfPunishedOrRewarded(environmentFeatureVector, predictedLabel)

		if isRewarded then

			reward = defaultReward

		else

			reward = defaultPunishment

		end

		table.insert(isRewardedArray, isRewarded)

		currentAverageAccuracy = getCurrentAverageAccuracy()

		if (currentAverageAccuracy ~= nil) then print("Accuracy: " .. currentAverageAccuracy .. "\tTime taken: " .. timeTaken) end

		task.wait(0.01)

	end

end

local function run()

	local Model = buildModel2()

	startEnvironment(Model)

end

run()
