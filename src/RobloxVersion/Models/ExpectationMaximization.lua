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

local IterativeMethodBaseModel = require(script.Parent.IterativeMethodBaseModel)

local ExpectationMaximizationModel = {}

ExpectationMaximizationModel.__index = ExpectationMaximizationModel

setmetatable(ExpectationMaximizationModel, IterativeMethodBaseModel)

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local defaultMaximumNumberOfIterations = 10

local defaultNumberOfClusters = math.huge

local defaultMode = "Hybrid"

local defaultEpsilon = math.pow(10, -9)

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

-- This function updates the model parameters based on the responsibility matrix

local function maximizationStep(featureMatrix, responsibilityMatrix, numberOfClusters, sumWeightMatrix, sumWeightXMatrix) -- data x features, data x clusters, clusters x 1, clusters x features 

	local numberOfData = #featureMatrix

	local numberOfFeatures = #featureMatrix[1]

	local piMatrix = AqwamTensorLibrary:sum(responsibilityMatrix, 1)

	piMatrix = AqwamTensorLibrary:divide(piMatrix, numberOfData)

	piMatrix = AqwamTensorLibrary:transpose(piMatrix)

	local responsibilitiesMatrixTransposed = AqwamTensorLibrary:transpose(responsibilityMatrix) -- clusters x data
	
	local subSumWeightMatrix = AqwamTensorLibrary:sum(responsibilitiesMatrixTransposed, 2) -- clusters x 1
	
	local subSumWeightXMatrix = AqwamTensorLibrary:dotProduct(responsibilitiesMatrixTransposed, featureMatrix) -- clusters x features
	
	sumWeightMatrix = AqwamTensorLibrary:add(sumWeightMatrix, subSumWeightMatrix) -- clusters x 1

	sumWeightXMatrix = AqwamTensorLibrary:add(sumWeightXMatrix, subSumWeightXMatrix) -- clusters x features

	local meanMatrix = AqwamTensorLibrary:divide(sumWeightXMatrix, sumWeightMatrix) -- clusters x features

	local varianceMatrix = AqwamTensorLibrary:createTensor({numberOfClusters, numberOfFeatures}, 0)

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

function ExpectationMaximizationModel:initializeMatrices(numberOfClusters, numberOfFeatures)
	
	local centroidMatrixDimensionSizeArray = {numberOfClusters, numberOfFeatures}
	
	local piMatrix = self:initializeMatrixBasedOnMode({numberOfClusters, 1})

	local meanMatrix = self:initializeMatrixBasedOnMode(centroidMatrixDimensionSizeArray)

	local varianceMatrix = self:initializeMatrixBasedOnMode(centroidMatrixDimensionSizeArray)

	local sumWeightMatrix = AqwamTensorLibrary:createTensor(centroidMatrixDimensionSizeArray)

	local sumWeightXMatrix = AqwamTensorLibrary:createTensor(centroidMatrixDimensionSizeArray)
	
	return piMatrix, meanMatrix, varianceMatrix, sumWeightMatrix, sumWeightXMatrix
	
end

function ExpectationMaximizationModel:getBayesianInformationCriterion(featureMatrix, numberOfClusters, epsilon)
	
	local numberOfData = #featureMatrix
	
	local numberOfFeatures = #featureMatrix[1]
	
	local piMatrix, meanMatrix, varianceMatrix, sumWeightMatrix, sumWeightXMatrix = self:initializeMatrices(numberOfClusters, numberOfFeatures)
	
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

	local logLikelihoodMatrix

	local sumLogLikelihood
	
	local cost
	
	if (not piMatrix) or (not meanMatrix) or (not varianceMatrix) or (not sumWeightMatrix) or (not sumWeightXMatrix) then
		
		if (numberOfClusters == math.huge) then 
			
			piMatrix, meanMatrix, varianceMatrix, sumWeightMatrix, sumWeightXMatrix = self:getBestMatrices(featureMatrix, epsilon) 
			
		else
			
			piMatrix, meanMatrix, varianceMatrix, sumWeightMatrix, sumWeightXMatrix = self:initializeMatrices(numberOfClusters, numberOfFeatures)
			
		end
		
	end

	repeat
		
		numberOfIterations = numberOfIterations + 1
		
		self:iterationWait()

		responsibilityMatrix = expectationStep(featureMatrix, piMatrix, meanMatrix, varianceMatrix, epsilon)

		piMatrix, meanMatrix, varianceMatrix, sumWeightMatrix, sumWeightXMatrix = maximizationStep(featureMatrix, responsibilityMatrix, numberOfClusters, sumWeightMatrix, sumWeightXMatrix)
		
		gaussianMatrix = calculateGaussianMatrix(featureMatrix, piMatrix, meanMatrix, varianceMatrix, epsilon)
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()
			
			logLikelihoodMatrix = AqwamTensorLibrary:applyFunction(math.log, gaussianMatrix)

			sumLogLikelihood = AqwamTensorLibrary:sum(logLikelihoodMatrix)

			table.insert(logLikelihoodArray, sumLogLikelihood)
			
			local logLikelihoodArrayLength = #logLikelihoodArray

			if (logLikelihoodArrayLength > 1) then

				cost = sumLogLikelihood - logLikelihoodArray[logLikelihoodArrayLength - 1] 

			else

				cost = -sumLogLikelihood

			end
			
		end)
		
		if (cost) then
			
			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)

		end

	until (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	if (cost == math.huge) then warn("The model diverged! Please repeat the experiment again or change the argument values.") end

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
