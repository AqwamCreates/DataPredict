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
	
	local probabilitiesMatrix = AqwamTensorLibrary:createTensor({#featureMatrix, numberOfClusters}, 0)
	
	for i = 1, #featureMatrix, 1 do

		local featureVector = {featureMatrix[i]}

		for j = 1, numberOfClusters, 1 do

			local weight = piMatrix[j][1]

			local meanVector = {meanMatrix[j]}

			local varianceVector = {varianceMatrix[j]}

			local probabilitiesVector = gaussian(featureVector, meanVector, varianceVector, epsilon)

			for i, probability in ipairs(probabilitiesVector[1]) do weight = weight * probability end

			probabilitiesMatrix[i][j] = weight

		end

	end
	
	return probabilitiesMatrix
	
end

function ExpectationMaximizationModel:initializeParameters(numberOfClusters, numberOfFeatures)

	local piMatrix = self:initializeMatrixBasedOnMode({numberOfClusters, 1})

	local meanMatrix = self:initializeMatrixBasedOnMode({numberOfClusters, numberOfFeatures})

	local varianceMatrix = self:initializeMatrixBasedOnMode({numberOfClusters, numberOfFeatures})

	return piMatrix, meanMatrix, varianceMatrix
	
end

local function expectationStep(featureMatrix, numberOfClusters, piMatrix, meanMatrix, varianceMatrix, epsilon)
	
	local numberOfData = #featureMatrix
	
	local responsibilityMatrix = calculateGaussianMatrix(featureMatrix, piMatrix, meanMatrix, varianceMatrix, epsilon) -- number of data x number of columns
	
	local responsibilitySumVector = AqwamTensorLibrary:sum(responsibilityMatrix, 1)
	
	local normalizedResponsibilityMatrix = AqwamTensorLibrary:divide(responsibilityMatrix, responsibilitySumVector)
	
	return normalizedResponsibilityMatrix
	
end

-- This function updates the model parameters based on the responsibility matrix

local function maximizationStep(featureMatrix, responsibilityMatrix, numberOfClusters) -- data x features, data x clusters

	local numberOfData = #featureMatrix

	local numberOfFeatures = #featureMatrix[1]

	local piMatrix = AqwamTensorLibrary:sum(responsibilityMatrix, 1)

	piMatrix = AqwamTensorLibrary:divide(piMatrix, numberOfData)

	piMatrix = AqwamTensorLibrary:transpose(piMatrix)

	local responsibilitiesMatrixTransposed = AqwamTensorLibrary:transpose(responsibilityMatrix) -- clusters x data

	local sumWeight = AqwamTensorLibrary:sum(responsibilitiesMatrixTransposed, 2) -- clusters x 1

	local sumWeightX = AqwamTensorLibrary:dotProduct(responsibilitiesMatrixTransposed, featureMatrix) -- clusters x features

	local meanMatrix = AqwamTensorLibrary:divide(sumWeightX, sumWeight) -- clusters x features

	local varianceMatrix = AqwamTensorLibrary:createTensor({numberOfClusters, numberOfFeatures}, 0)

	for i = 1, numberOfClusters, 1 do

		local meanVector = {meanMatrix[i]}

		local thisStandardDeviationMatrix = AqwamTensorLibrary:subtract(featureMatrix, meanVector)

		local thisVariationMatrix = AqwamTensorLibrary:power(thisStandardDeviationMatrix, 2)

		local thisSumVariationMatrix = AqwamTensorLibrary:sum(thisVariationMatrix, 1)

		varianceMatrix[i] = thisSumVariationMatrix[1]

	end

	varianceMatrix = AqwamTensorLibrary:divide(varianceMatrix, sumWeight)

	return piMatrix, meanMatrix, varianceMatrix

end

function ExpectationMaximizationModel:getBayesianInformationCriterion(featureMatrix, numberOfClusters, epsilon)
	
	local numberOfFeatures = #featureMatrix[1]
	
	local piMatrix, meanMatrix, varianceMatrix = self:initializeParameters(numberOfClusters, numberOfFeatures)
	
	local responsibilityMatrix = expectationStep(featureMatrix, numberOfClusters, piMatrix, meanMatrix, varianceMatrix, epsilon)
	
	local piMatrix, meanMatrix, varianceMatrix = maximizationStep(featureMatrix, responsibilityMatrix, numberOfClusters)
	
	local gaussianMatrix = calculateGaussianMatrix(featureMatrix, piMatrix, meanMatrix, varianceMatrix, epsilon)
	
	local logLikelihood = AqwamTensorLibrary:logarithm(gaussianMatrix)

	local sumLogLikelihood = AqwamTensorLibrary:sum(logLikelihood)
	
	local numberOfData = #featureMatrix
	
	local k = (numberOfClusters - 1) + numberOfClusters * numberOfFeatures * 2
	
	local bayesianInformationCriterion = (k * math.log(numberOfData)) - (2 * sumLogLikelihood)
	
	return bayesianInformationCriterion
	
end

function ExpectationMaximizationModel:fetchBestNumberOfClusters(featureMatrix, epsilon)
	
	local numberOfClusters = 2
	
	local bestBayesianInformationCriterion = math.huge
	
	local bestNumberOfClusters = numberOfClusters

	while true do
		
		local bayesianInformationCriterion = self:getBayesianInformationCriterion(featureMatrix, numberOfClusters, epsilon)

		if (bayesianInformationCriterion < bestBayesianInformationCriterion) then
			
			bestBayesianInformationCriterion = bayesianInformationCriterion
			
			bestNumberOfClusters = numberOfClusters
			
		else
			
			break
			
		end

		numberOfClusters = numberOfClusters + 1
		
	end

	return bestNumberOfClusters
	
end

function ExpectationMaximizationModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaximumNumberOfIterations

	local NewExpectationMaximizationModel = IterativeMethodBaseModel.new(parameterDictionary)

	setmetatable(NewExpectationMaximizationModel, ExpectationMaximizationModel)
	
	NewExpectationMaximizationModel:setName("ExpectationMaximization")
	
	NewExpectationMaximizationModel.numberOfClusters = parameterDictionary.numberOfClusters or defaultNumberOfClusters

	NewExpectationMaximizationModel.epsilon = parameterDictionary.epsilon or defaultEpsilon
	
	return NewExpectationMaximizationModel
end

function ExpectationMaximizationModel:train(featureMatrix)
	
	local piMatrix
	
	local meanMatrix
	
	local varianceMatrix
	
	local responsibilityMatrix
	
	local gaussianMatrix

	local costArray = {}
	
	local logLikelihoodArray = {}
	
	local cost
	
	local numberOfIterations = 0

	local logLikelihood
	
	local sumLogLikelihood
	
	local numberOfFeatures = #featureMatrix[1]
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local numberOfClusters = self.numberOfClusters
	
	local epsilon = self.epsilon
	
	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		piMatrix, meanMatrix, varianceMatrix = table.unpack(ModelParameters)

		if (#featureMatrix[1] ~= #meanMatrix[1]) then error("The number of features are not the same as the model parameters!") end
		
	else
		
		if (numberOfClusters == math.huge) then
			
			numberOfClusters = self:fetchBestNumberOfClusters(featureMatrix, epsilon)
			
			self.numberOfClusters = numberOfClusters
			
		end
		
		piMatrix, meanMatrix, varianceMatrix = self:initializeParameters(numberOfClusters, numberOfFeatures) 

	end

	repeat
		
		numberOfIterations += 1
		
		self:iterationWait()

		responsibilityMatrix = expectationStep(featureMatrix, numberOfClusters, piMatrix, meanMatrix, varianceMatrix, epsilon)

		piMatrix, meanMatrix, varianceMatrix = maximizationStep(featureMatrix, responsibilityMatrix, numberOfClusters)
		
		gaussianMatrix = calculateGaussianMatrix(featureMatrix, piMatrix, meanMatrix, varianceMatrix, epsilon)
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()
			
			logLikelihood = AqwamTensorLibrary:applyFunction(math.log, gaussianMatrix)

			sumLogLikelihood = AqwamTensorLibrary:sum(logLikelihood)

			table.insert(logLikelihoodArray, sumLogLikelihood)

			if (#logLikelihoodArray > 1) then

				cost = sumLogLikelihood - logLikelihoodArray[#logLikelihoodArray - 1] 

			else

				cost = -sumLogLikelihood

			end
			
		end)
		
		if (cost) then
			
			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)
			
			if (cost ~= cost) then error("Too much variance in the data! Please change the argument values.") end

		end

	until (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	self.ModelParameters = {piMatrix, meanMatrix, varianceMatrix}

	return costArray

end

function ExpectationMaximizationModel:predict(featureMatrix, returnOriginalOutput)
	
	local numberOfFeatures = #featureMatrix
	
	local piMatrix, meanMatrix, varianceMatrix = table.unpack(self.ModelParameters)

	local gaussianMatrix = calculateGaussianMatrix(featureMatrix, piMatrix, meanMatrix, varianceMatrix, self.epsilon)
	
	if (returnOriginalOutput == true) then return gaussianMatrix end
	
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
