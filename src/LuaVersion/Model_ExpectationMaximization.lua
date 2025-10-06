--[[

	--------------------------------------------------------------------

	Aqwam's Machine, Deep And Reinforcement Learning Library (DataPredict)

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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local IterativeMethodBaseModel = require("Model_IterativeMethodBaseModel")

local ExpectationMaximizationModel = {}

ExpectationMaximizationModel.__index = ExpectationMaximizationModel

setmetatable(ExpectationMaximizationModel, IterativeMethodBaseModel)

local defaultMaximumNumberOfIterations = 10

local defaultNumberOfClusters = math.huge

local defaultMode = "Hybrid"

local defaultEpsilon = math.pow(10, -16)

local defaultSetInitialCentroidsOnDataPoints = true

local defaultSetTheCentroidsDistanceFarthest = true

local defaultDistanceFunction = "Euclidean"

local distanceFunctionList = {

	["Manhattan"] = function (x1, x2)

		local part1 = AqwamTensorLibrary:subtract(x1, x2)

		part1 = AqwamTensorLibrary:applyFunction(math.abs, part1)

		local distance = AqwamTensorLibrary:sum(part1)

		return distance 

	end,

	["Euclidean"] = function (x1, x2)

		local part1 = AqwamTensorLibrary:subtract(x1, x2)

		local part2 = AqwamTensorLibrary:power(part1, 2)

		local part3 = AqwamTensorLibrary:sum(part2)

		local distance = math.sqrt(part3)

		return distance 

	end,

	["Cosine"] = function(x1, x2)

		local dotProductedX = AqwamTensorLibrary:dotProduct(x1, AqwamTensorLibrary:transpose(x2))

		local x1MagnitudePart1 = AqwamTensorLibrary:power(x1, 2)

		local x1MagnitudePart2 = AqwamTensorLibrary:sum(x1MagnitudePart1)

		local x1Magnitude = math.sqrt(x1MagnitudePart2)

		local x2MagnitudePart1 = AqwamTensorLibrary:power(x2, 2)

		local x2MagnitudePart2 = AqwamTensorLibrary:sum(x2MagnitudePart1)

		local x2Magnitude = math.sqrt(x2MagnitudePart2)

		local normX = x1Magnitude * x2Magnitude

		local similarity = dotProductedX / normX

		local cosineDistance = 1 - similarity

		return cosineDistance

	end,

}

local function gaussian(featureVector, meanVector, varianceVector, epsilon)
	
	local exponentStep1 = AqwamTensorLibrary:subtract(featureVector, meanVector)

	local exponentStep2 = AqwamTensorLibrary:power(exponentStep1, 2)

	local exponentStep3 = AqwamTensorLibrary:divide(exponentStep2, varianceVector)

	local exponentStep4 = AqwamTensorLibrary:multiply(-0.5, exponentStep3)

	local exponentWithTerms = AqwamTensorLibrary:exponent(exponentStep4)
	
	local standardDeviationVector = AqwamTensorLibrary:power(varianceVector, 0.5)

	local divisorPart1 = AqwamTensorLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))
	
	local divisor = AqwamTensorLibrary:add(divisorPart1, epsilon)

	local gaussianDensity = AqwamTensorLibrary:divide(exponentWithTerms, divisor)
	
	return gaussianDensity

end

local function calculateGaussianMatrix(featureMatrix, piMatrix, meanMatrix, varianceMatrix, epsilon)
	
	local numberOfClusters = #meanMatrix
	
	local probabilityMatrix = AqwamTensorLibrary:createTensor({#featureMatrix, numberOfClusters}, 0)
	
	for i = 1, #featureMatrix, 1 do

		local featureVector = {featureMatrix[i]}

		for j = 1, numberOfClusters, 1 do

			local weight = piMatrix[j][1]

			local meanVector = {meanMatrix[j]}

			local varianceVector = {varianceMatrix[j]}

			local probabilitiesVector = gaussian(featureVector, meanVector, varianceVector, epsilon)

			for i, probability in ipairs(probabilitiesVector[1]) do weight = weight * probability end

			probabilityMatrix[i][j] = weight

		end

	end
	
	return probabilityMatrix
	
end

local function expectationStep(featureMatrix, piMatrix, meanMatrix, varianceMatrix, epsilon)
	
	local responsibilityMatrix = calculateGaussianMatrix(featureMatrix, piMatrix, meanMatrix, varianceMatrix, epsilon) -- number of data x number of columns
	
	local responsibilitySumVector = AqwamTensorLibrary:sum(responsibilityMatrix, 1)
	
	local normalizedResponsibilityMatrix = AqwamTensorLibrary:divide(responsibilityMatrix, responsibilitySumVector)
	
	return normalizedResponsibilityMatrix
	
end

local function maximizationStep(featureMatrix, responsibilityMatrix, numberOfClusters, sumWeightMatrix, sumWeightXMatrix) -- data x features, data x clusters, clusters x 1, clusters x features

	local piMatrix = AqwamTensorLibrary:sum(responsibilityMatrix, 1)
	
	local piSum = AqwamTensorLibrary:sum(piMatrix)

	piMatrix = AqwamTensorLibrary:transpose(piMatrix)
	
	piMatrix = AqwamTensorLibrary:divide(piMatrix, piSum)

	local responsibilitiesMatrixTransposed = AqwamTensorLibrary:transpose(responsibilityMatrix) -- clusters x data
	
	local subSumWeightMatrix = AqwamTensorLibrary:sum(responsibilitiesMatrixTransposed, 2) -- clusters x 1
	
	local subSumWeightXMatrix = AqwamTensorLibrary:dotProduct(responsibilitiesMatrixTransposed, featureMatrix) -- clusters x features
	
	sumWeightMatrix = AqwamTensorLibrary:add(sumWeightMatrix, subSumWeightMatrix) -- clusters x 1

	sumWeightXMatrix = AqwamTensorLibrary:add(sumWeightXMatrix, subSumWeightXMatrix) -- clusters x features

	local meanMatrix = AqwamTensorLibrary:divide(sumWeightXMatrix, sumWeightMatrix) -- clusters x features

	local varianceMatrix = AqwamTensorLibrary:createTensor({numberOfClusters, #featureMatrix[1]}, 0)

	for i = 1, numberOfClusters, 1 do

		local meanVector = {meanMatrix[i]}

		local thisStandardDeviationMatrix = AqwamTensorLibrary:subtract(featureMatrix, meanVector)

		local thisVariationMatrix = AqwamTensorLibrary:power(thisStandardDeviationMatrix, 2)

		local thisSumVariationMatrix = AqwamTensorLibrary:sum(thisVariationMatrix, 1)

		varianceMatrix[i] = thisSumVariationMatrix[1]

	end

	varianceMatrix = AqwamTensorLibrary:divide(varianceMatrix, sumWeightMatrix)

	return piMatrix, meanMatrix, varianceMatrix, sumWeightMatrix, sumWeightXMatrix

end

local function calculateCost(gaussianMatrix, epsilon)
	
	local clampedGaussianMatrix = AqwamTensorLibrary:applyFunction(math.max, gaussianMatrix, {{epsilon}})
	
	local logLikelihoodMatrix = AqwamTensorLibrary:applyFunction(math.log, clampedGaussianMatrix)

	local sumLogLikelihood = AqwamTensorLibrary:sum(logLikelihoodMatrix)
	
	return -sumLogLikelihood
	
end

local function chooseFarthestCentroidFromDatasetDistanceMatrix(distanceMatrix, blacklistedDataIndexArray)

	local dataIndex

	local maxDistance = -math.huge

	for row = 1, #distanceMatrix, 1 do

		if table.find(blacklistedDataIndexArray, row) then continue end

		local totalDistance = 0

		for column = 1, #distanceMatrix[1], 1 do

			totalDistance = totalDistance + distanceMatrix[row][column]

		end

		if (totalDistance < maxDistance) then continue end

		maxDistance = totalDistance
		dataIndex = row

	end

	return dataIndex

end

local function createDistanceMatrix(matrix1, matrix2, distanceFunction)

	local numberOfData1 = #matrix1

	local numberOfData2 = #matrix2

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData1, numberOfData2})

	for matrix1Index = 1, numberOfData1, 1 do

		for matrix2Index = 1, numberOfData2, 1 do

			distanceMatrix[matrix1Index][matrix2Index] = distanceFunction({matrix1[matrix1Index]}, {matrix2[matrix2Index]})

		end

	end

	return distanceMatrix

end

local function chooseFarthestCentroids(featureMatrix, numberOfClusters, distanceFunction)

	local centroidMatrix = {}

	local dataIndexArray = {}

	local dataIndex

	local distanceMatrix = createDistanceMatrix(featureMatrix, featureMatrix, distanceFunction)

	repeat

		dataIndex = chooseFarthestCentroidFromDatasetDistanceMatrix(distanceMatrix, dataIndexArray)

		table.insert(dataIndexArray, dataIndex)

	until (#dataIndexArray == numberOfClusters)

	for row = 1, numberOfClusters, 1 do

		dataIndex = dataIndexArray[row]

		table.insert(centroidMatrix, featureMatrix[dataIndex])

	end

	return centroidMatrix

end

local function chooseRandomCentroids(featureMatrix, numberOfClusters)

	local modelParameters = {}

	local numberOfRows = #featureMatrix

	local randomRow

	local selectedRows = {}

	local hasANewRandomRowChosen

	for cluster = 1, numberOfClusters, 1 do

		repeat

			randomRow = Random.new():NextInteger(1, numberOfRows)

			hasANewRandomRowChosen = not (table.find(selectedRows, randomRow))

			if hasANewRandomRowChosen then

				table.insert(selectedRows, randomRow)
				modelParameters[cluster] = featureMatrix[randomRow]

			end

		until hasANewRandomRowChosen

	end

	return modelParameters

end

function ExpectationMaximizationModel:initializeCentroids(featureMatrix, numberOfClusters)
	
	local setInitialCentroidsOnDataPoints = self.setInitialCentroidsOnDataPoints
	
	local setTheCentroidsDistanceFarthest = self.setTheCentroidsDistanceFarthest
	
	if (setInitialCentroidsOnDataPoints) and (numberOfClusters == 1) then

		return AqwamTensorLibrary:mean(featureMatrix, 1)

	elseif (setInitialCentroidsOnDataPoints) and (setTheCentroidsDistanceFarthest) then
		
		local distanceFunctionToApply = distanceFunctionList[self.distanceFunction]
		
		if (not distanceFunctionToApply) then error("Unknown distance function.") end

		return chooseFarthestCentroids(featureMatrix, numberOfClusters, distanceFunctionToApply)

	elseif (setInitialCentroidsOnDataPoints) and (not setTheCentroidsDistanceFarthest) then

		return chooseRandomCentroids(featureMatrix, numberOfClusters)

	else

		return self:initializeMatrixBasedOnMode({numberOfClusters, #featureMatrix[1]})

	end

end

function ExpectationMaximizationModel:initializeMatrices(featureMatrix, numberOfClusters, numberOfFeatures)
	
	local centroidMatrixDimensionSizeArray = {numberOfClusters, numberOfFeatures}
	
	local piMatrix = AqwamTensorLibrary:createRandomUniformTensor({numberOfClusters, 1})
	
	local sumPi = AqwamTensorLibrary:sum(piMatrix)

	local meanMatrix = self:initializeCentroids(featureMatrix, numberOfClusters)

	local varianceMatrix = AqwamTensorLibrary:createRandomUniformTensor(centroidMatrixDimensionSizeArray, 0, 1)

	local sumWeightMatrix = AqwamTensorLibrary:createTensor(centroidMatrixDimensionSizeArray)

	local sumWeightXMatrix = AqwamTensorLibrary:createTensor(centroidMatrixDimensionSizeArray)
	
	piMatrix = AqwamTensorLibrary:divide(piMatrix, sumPi)
	
	return piMatrix, meanMatrix, varianceMatrix, sumWeightMatrix, sumWeightXMatrix
	
end

function ExpectationMaximizationModel:getBayesianInformationCriterion(featureMatrix, numberOfClusters, epsilon)
	
	local numberOfData = #featureMatrix
	
	local numberOfFeatures = #featureMatrix[1]
	
	local piMatrix, meanMatrix, varianceMatrix, sumWeightMatrix, sumWeightXMatrix = self:initializeMatrices(featureMatrix, numberOfClusters, numberOfFeatures)
	
	local responsibilityMatrix = expectationStep(featureMatrix, piMatrix, meanMatrix, varianceMatrix, epsilon)
	
	piMatrix, meanMatrix, varianceMatrix, sumWeightMatrix, sumWeightXMatrix = maximizationStep(featureMatrix, responsibilityMatrix, numberOfClusters, sumWeightMatrix, sumWeightXMatrix)
	
	local gaussianMatrix = calculateGaussianMatrix(featureMatrix, piMatrix, meanMatrix, varianceMatrix, epsilon)
	
	local logLikelihood = AqwamTensorLibrary:logarithm(gaussianMatrix)

	local sumLogLikelihood = AqwamTensorLibrary:sum(logLikelihood)
	
	local k = (numberOfClusters - 1) + (numberOfClusters * numberOfFeatures * 2)
	
	local bayesianInformationCriterion = (k * math.log(numberOfData)) - (2 * sumLogLikelihood)
	
	return bayesianInformationCriterion, piMatrix, meanMatrix, varianceMatrix, sumWeightMatrix, sumWeightXMatrix
	
end

function ExpectationMaximizationModel:getBestMatrices(featureMatrix, epsilon)
	
	local numberOfFeatures = #featureMatrix[1]
	
	local numberOfClusters = 1
	
	local bestBayesianInformationCriterion = math.huge
	
	local bestNumberOfClusters = numberOfClusters
	
	local bayesianInformationCriterion
	
	local piMatrix
	
	local meanMatrix
	
	local varianceMatrix
	
	local sumWeightMatrix
	
	local sumWeightXMatrix
	
	local bestPiMatrix

	local bestMeanMatrix

	local bestVarianceMatrix

	local bestSumWeightMatrix

	local bestSumWeightXMatrix

	while true do
		
		bayesianInformationCriterion, piMatrix, meanMatrix, varianceMatrix, sumWeightMatrix, sumWeightXMatrix = self:getBayesianInformationCriterion(featureMatrix, numberOfClusters, epsilon)

		if (bayesianInformationCriterion < bestBayesianInformationCriterion) then
			
			bestBayesianInformationCriterion = bayesianInformationCriterion
			
			bestNumberOfClusters = numberOfClusters
			
			bestPiMatrix = piMatrix
			
			bestMeanMatrix = meanMatrix
			
			bestVarianceMatrix = varianceMatrix
			
			bestSumWeightMatrix = sumWeightMatrix
			
			bestSumWeightXMatrix = sumWeightXMatrix
			
		else
			
			break
			
		end

		numberOfClusters = numberOfClusters + 1
		
	end

	return bestPiMatrix, bestMeanMatrix, bestVarianceMatrix, bestSumWeightMatrix, bestSumWeightXMatrix
	
end

function ExpectationMaximizationModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewExpectationMaximizationModel = IterativeMethodBaseModel.new(parameterDictionary)

	setmetatable(NewExpectationMaximizationModel, ExpectationMaximizationModel)
	
	NewExpectationMaximizationModel:setName("ExpectationMaximization")
	
	NewExpectationMaximizationModel.numberOfClusters = parameterDictionary.numberOfClusters or defaultNumberOfClusters
	
	NewExpectationMaximizationModel.mode = parameterDictionary.mode or defaultMode

	NewExpectationMaximizationModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	NewExpectationMaximizationModel.setInitialCentroidsOnDataPoints =  NewExpectationMaximizationModel:getValueOrDefaultValue(parameterDictionary.setInitialCentroidsOnDataPoints, defaultSetInitialCentroidsOnDataPoints)

	NewExpectationMaximizationModel.setTheCentroidsDistanceFarthest = NewExpectationMaximizationModel:getValueOrDefaultValue(parameterDictionary.setTheCentroidsDistanceFarthest, defaultSetTheCentroidsDistanceFarthest)
	
	NewExpectationMaximizationModel.distanceFunction = NewExpectationMaximizationModel:getValueOrDefaultValue(parameterDictionary.distanceFunction, defaultDistanceFunction)
	
	return NewExpectationMaximizationModel
end

function ExpectationMaximizationModel:train(featureMatrix)
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations

	local numberOfClusters = self.numberOfClusters
	
	local mode = self.mode

	local epsilon = self.epsilon

	local ModelParameters = self.ModelParameters or {}
	
	local numberOfFeatures = #featureMatrix[1]
	
	local piMatrix = ModelParameters[1]

	local meanMatrix = ModelParameters[2]

	local varianceMatrix = ModelParameters[3]
	
	local sumWeightMatrix = ModelParameters[4]
	
	local sumWeightXMatrix = ModelParameters[5]
	
	if (mode == "Hybrid") then
		
		mode = (piMatrix and meanMatrix and varianceMatrix and sumWeightMatrix and sumWeightXMatrix and "Online") or "Offline"		
		
	end
	
	if (mode == "Offline") then
		
		piMatrix = nil
		
		meanMatrix = nil
		
		varianceMatrix = nil
		
		sumWeightMatrix = nil
		
		sumWeightXMatrix = nil
		
	end
	
	local logLikelihoodArray = {}
	
	local costArray = {}
	
	local numberOfIterations = 0
	
	local responsibilityMatrix
	
	local gaussianMatrix
	
	local cost
	
	if (not piMatrix) or (not meanMatrix) or (not varianceMatrix) or (not sumWeightMatrix) or (not sumWeightXMatrix) then
		
		if (numberOfClusters == math.huge) then 
			
			piMatrix, meanMatrix, varianceMatrix, sumWeightMatrix, sumWeightXMatrix = self:getBestMatrices(featureMatrix, epsilon)
			
		else
			
			piMatrix, meanMatrix, varianceMatrix, sumWeightMatrix, sumWeightXMatrix = self:initializeMatrices(featureMatrix, numberOfClusters, numberOfFeatures)
			
		end
		
	end
	
	numberOfClusters = #piMatrix -- This should be outside because nothing is replacing infinite number of clusters when it is given as a parameter after the first training.
	
	repeat
		
		numberOfIterations = numberOfIterations + 1
		
		self:iterationWait()

		responsibilityMatrix = expectationStep(featureMatrix, piMatrix, meanMatrix, varianceMatrix, epsilon)

		piMatrix, meanMatrix, varianceMatrix, sumWeightMatrix, sumWeightXMatrix = maximizationStep(featureMatrix, responsibilityMatrix, numberOfClusters, sumWeightMatrix, sumWeightXMatrix)
		
		gaussianMatrix = calculateGaussianMatrix(featureMatrix, piMatrix, meanMatrix, varianceMatrix, epsilon)
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()
			
			return calculateCost(gaussianMatrix, epsilon)
			
		end)
		
		if (cost) then
			
			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end

	until (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end
	
	-- We're just normalizing here to so that the sumWeightMatrix and sumWeightMatrix values doesn't go so big to the point of numerical overflow.
	
	local normalizationDenominator = AqwamTensorLibrary:sum(sumWeightMatrix)

	sumWeightMatrix = AqwamTensorLibrary:divide(sumWeightMatrix, normalizationDenominator)

	sumWeightXMatrix = AqwamTensorLibrary:divide(sumWeightXMatrix, normalizationDenominator)
	
	self.ModelParameters = {piMatrix, meanMatrix, varianceMatrix, sumWeightMatrix, sumWeightXMatrix}

	return costArray

end

function ExpectationMaximizationModel:predict(featureMatrix, returnOriginalOutput)
	
	local numberOfFeatures = #featureMatrix
	
	local piMatrix, meanMatrix, varianceMatrix = table.unpack(self.ModelParameters)

	local gaussianMatrix = calculateGaussianMatrix(featureMatrix, piMatrix, meanMatrix, varianceMatrix, self.epsilon)
	
	if (returnOriginalOutput) then return gaussianMatrix end
	
	local selectedClustersVector = AqwamTensorLibrary:createTensor({numberOfFeatures, 1})

	local probabilityVector = AqwamTensorLibrary:createTensor({numberOfFeatures, 1})
	
	for dataIndex, gausssianVector in ipairs(gaussianMatrix) do
		
		local selectedCluster
		
		local highestWeight = -math.huge
		
		for clusterNumber, weight in ipairs(gausssianVector) do
			
			if (weight > highestWeight) then
				
				selectedCluster = clusterNumber

				highestWeight = weight
				
			end
				
		end
		
		selectedClustersVector[dataIndex][1] = selectedCluster
		
		probabilityVector[dataIndex][1] = highestWeight
		
	end
	
	return selectedClustersVector, probabilityVector
	
end

return ExpectationMaximizationModel
