local BaseModel = require(script.Parent.BaseModel)

local ExpectationMaximizationModel = {}

ExpectationMaximizationModel.__index = ExpectationMaximizationModel

setmetatable(ExpectationMaximizationModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultEpsilon = 1 * math.exp(-10)

local defaultMaxNumberOfIterations = 10

local defaultNumberOfClusters = nil

local defaultTargetCost = 0

local function gaussian(x, mean, variance, epsilon)

	local z = -(x - mean)^2 / (2 * (variance + epsilon))

	local exponentPart = math.exp(z)

	local coefficient = 1 / math.sqrt(2 * math.pi * (variance + epsilon))

	local result = coefficient * exponentPart

	return result

end

-- This function initializes the model parameters randomly
local function initializeParameters(featureMatrix, numberOfClusters)

	local numberOfFeatures = #featureMatrix[1]

	local piTable = {}

	local meanMatrix = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfFeatures, numberOfClusters)

	local varianceMatrix = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfFeatures, numberOfClusters)

	varianceMatrix = AqwamMatrixLibrary:applyFunction(math.abs, varianceMatrix)

	-- Randomly initialize mixing coefficients
	for i = 1, numberOfClusters do

		piTable[i] = Random.new():NextNumber()

	end

	return piTable, meanMatrix, varianceMatrix
end

-- This function computes the responsibility matrix for each featureMatrix point
local function eStep(featureMatrix, numberOfClusters, piTable, meanMatrix, varianceMatrix, epsilon)

	local numberOfData = #featureMatrix
	local responsibilities = AqwamMatrixLibrary:createMatrix(numberOfData, numberOfClusters)

	-- Compute the responsibility matrix for each featureMatrix point
	for k = 1, numberOfData do

		local totalWeight = 0

		for i = 1, numberOfClusters, 1 do

			local weight = piTable[i]

			for j = 1, #featureMatrix[k] do

				weight = weight * gaussian(featureMatrix[k][j], meanMatrix[j][i], varianceMatrix[j][i], epsilon)

			end

			responsibilities[k][i] = weight

			totalWeight = totalWeight + weight

		end

		-- Normalize the responsibilities so that they sum up to 1
		for i = 1, numberOfClusters do

			responsibilities[k][i] = responsibilities[k][i] / totalWeight

		end

	end

	return responsibilities
end

-- This function updates the model parameters based on the responsibility matrix
local function mStep(featureMatrix, responsibilities, numberOfClusters)

	local numberOfData = #featureMatrix

	local numberOfFeatures = #featureMatrix[1]

	local piTable = {}

	local meanMatrix = {}

	local varianceMatrix = {}

	-- Update mixing coefficients
	for i = 1, numberOfClusters do

		piTable[i] = 0

		for k = 1, numberOfData do

			piTable[i] = piTable[i] + responsibilities[k][i]

		end

		piTable[i] = piTable[i] / numberOfData

	end

	-- Update meanMatrix vectors and varianceMatrixs
	for j = 1, numberOfFeatures do

		meanMatrix[j] = {}

		varianceMatrix[j] = {}

		for i = 1, numberOfClusters do

			local sumWeight = 0

			local sumWeightX = 0

			for k = 1, numberOfData do

				sumWeight += responsibilities[k][i]

				sumWeightX +=  responsibilities[k][i] * featureMatrix[k][j]

			end

			meanMatrix[j][i] = sumWeightX / sumWeight

			varianceMatrix[j][i] = 0

			for k = 1, numberOfData do

				varianceMatrix[j][i] += responsibilities[k][i] * (featureMatrix[k][j] - meanMatrix[j][i])^2

			end

			varianceMatrix[j][i] = varianceMatrix[j][i] / sumWeight

		end

	end

	return piTable, meanMatrix, varianceMatrix

end

local function getBayesianInformationCriterion(featureMatrix, epsilon, numberOfClusters)
	
	local piTable, meanMatrix, varianceMatrix = initializeParameters(featureMatrix, numberOfClusters)
	
	local responsibilities = eStep(featureMatrix, numberOfClusters, piTable, meanMatrix, varianceMatrix, epsilon)
	
	local piTable, meanMatrix, varianceMatrix = mStep(featureMatrix, responsibilities, numberOfClusters)

	local likelihood = 0
	
	for k = 1, #featureMatrix do
		
		local data_likelihood = 0
		
		for i = 1, numberOfClusters do
			
			local weight = piTable[i]
			
			for j = 1, #featureMatrix[k] do
				
				weight = weight * gaussian(featureMatrix[k][j], meanMatrix[j][i], varianceMatrix[j][i], epsilon)
				
			end
			
			data_likelihood = data_likelihood + weight
			
		end
		
		likelihood = likelihood + math.log(data_likelihood)
		
	end

	local freeParameters = numberOfClusters * (#featureMatrix[1] + 1) * 2 -- number of mean, variance and mixture weights for each cluster
	
	local numberOfData = #featureMatrix
	
	local bayesianInformationCriterion = likelihood - (0.5 * freeParameters * math.log(numberOfData))
	
	return bayesianInformationCriterion
	
end

local function fetchBestNumberOfClusters(featureMatrix, epsilon, targetCost)
	
	local numberOfClusters = 2 -- Start with two clusters
	
	local bestBayesianInformationCriterion = -math.huge
	
	local bestNumberOfClusters = numberOfClusters

	while true do
		
		local bayesianInformationCriterion = getBayesianInformationCriterion(featureMatrix, epsilon, numberOfClusters)

		if (bayesianInformationCriterion > bestBayesianInformationCriterion) then
			
			bestBayesianInformationCriterion = bayesianInformationCriterion
			
			bestNumberOfClusters = numberOfClusters
			
		else
			
			break
			
		end

		numberOfClusters = numberOfClusters + 1
		
	end

	return bestNumberOfClusters
	
end

-- This function trains the EM model


function ExpectationMaximizationModel.new(maxNumberOfIterations, numberOfClusters, epsilon, targetCost)

	local NewExpectationMaximizationModel = BaseModel.new()

	setmetatable(NewExpectationMaximizationModel, ExpectationMaximizationModel)

	NewExpectationMaximizationModel.numberOfClusters = numberOfClusters or defaultNumberOfClusters

	NewExpectationMaximizationModel.epsilon = epsilon or defaultEpsilon

	NewExpectationMaximizationModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewExpectationMaximizationModel.targetCost = maxNumberOfIterations or defaultTargetCost

	return NewExpectationMaximizationModel
end

function ExpectationMaximizationModel:setParameters(maxNumberOfIterations, numberOfClusters, epsilon, targetCost)

	self.numberOfClusters = numberOfClusters or self.numberOfClusters

	self.epsilon = epsilon or self.epsilon

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.targetCost = targetCost

end


function ExpectationMaximizationModel:train(featureMatrix)
	
	local piTable
	
	local meanMatrix
	
	local varianceMatrix

	local costArray = {}
	local cost = math.huge
	local numberOfIterations = 0
	local meanMatrix

	if (self.ModelParameters) then

		piTable, meanMatrix, varianceMatrix = unpack(self.ModelParameters)

		if (#featureMatrix[1] ~= #meanMatrix[1]) then error("The number of features are not the same as the model parameters!") end
		
	else
		
		self.numberOfClusters = self.numberOfClusters or fetchBestNumberOfClusters(featureMatrix, self.epsilon, self.targetCost)
		
		piTable, meanMatrix, varianceMatrix = initializeParameters(featureMatrix, self.numberOfClusters)

	end

	local previousLikelihood

	local likelihood = 0

	repeat

		numberOfIterations += 1

		previousLikelihood = likelihood

		local responsibilities = eStep(featureMatrix, self.numberOfClusters, piTable, meanMatrix, varianceMatrix, self.epsilon)

		piTable, meanMatrix, varianceMatrix = mStep(featureMatrix, responsibilities, self.numberOfClusters)

		likelihood = 0

		for k = 1, #featureMatrix, 1 do

			local featureMatrixLikelihood = 0

			for i = 1, self.numberOfClusters, 1 do

				local weight = piTable[i]

				for j = 1, #featureMatrix[k], 1 do
					

					weight *= gaussian(featureMatrix[k][j], meanMatrix[j][i], varianceMatrix[j][i], self.epsilon)

				end

				featureMatrixLikelihood = featureMatrixLikelihood + weight

			end

			likelihood = likelihood + math.log(featureMatrixLikelihood)
			
		end

		cost = math.abs(likelihood - previousLikelihood)

		table.insert(costArray, cost)

		self:printCostAndNumberOfIterations(cost, numberOfIterations)

	until (numberOfIterations == self.maxNumberOfIterations) or (cost <= self.targetCost)

	self.ModelParameters = {piTable, meanMatrix, varianceMatrix}

	return costArray

end

function ExpectationMaximizationModel:predict(featureVector)

	local piTable, meanMatrix, varianceMatrix = unpack(self.ModelParameters)

	local numberOfClusters = self.numberOfClusters

	local probabilityVector
	local selectedCluster

	for k = 1, #featureVector, 1 do

		local max_weight = -math.huge

		local max_cluster = 0

		for i = 1, numberOfClusters, 1 do

			local weight = piTable[i]

			for j = 1, #featureVector[k] do

				weight *= gaussian(featureVector[k][j], meanMatrix[j][i], varianceMatrix[j][i], self.epsilon)

			end

			if weight > max_weight then

				probabilityVector = weight

				selectedCluster = i

			end

		end

	end

	return selectedCluster, probabilityVector
end

return ExpectationMaximizationModel
