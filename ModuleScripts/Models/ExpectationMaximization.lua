local BaseModel = require(script.Parent.BaseModel)

local ExpectationMaximizationModel = {}

ExpectationMaximizationModel.__index = ExpectationMaximizationModel

setmetatable(ExpectationMaximizationModel, BaseModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local defaultEpsilon = math.pow(10, -9)

local defaultMaxNumberOfIterations = 10

local defaultNumberOfClusters = math.huge

local defaultTargetCost = 0

local function gaussian(featureVector, meanVector, varianceVector, epsilon)
	
	local exponentStep1 = AqwamMatrixLibrary:subtract(featureVector, meanVector)

	local exponentStep2 = AqwamMatrixLibrary:power(exponentStep1, 2)

	local exponentStep3 = AqwamMatrixLibrary:divide(exponentStep2, varianceVector)

	local exponentStep4 = AqwamMatrixLibrary:multiply(-0.5, exponentStep3)

	local exponentWithTerms = AqwamMatrixLibrary:applyFunction(math.exp, exponentStep4)
	
	local standardDeviationVector = AqwamMatrixLibrary:applyFunction(math.sqrt, varianceVector)

	local divisorPart1 = AqwamMatrixLibrary:multiply(standardDeviationVector, math.sqrt(2 * math.pi))
	
	local divisor = AqwamMatrixLibrary:add(divisorPart1, epsilon)

	local gaussianDensity = AqwamMatrixLibrary:divide(exponentWithTerms, divisor)
	
	return gaussianDensity

end

local function calculateGaussianMatrix(featureMatrix, piMatrix, meanMatrix, varianceMatrix, epsilon)
	
	local numberOfClusters = #meanMatrix
	
	local probabilitiesMatrix = AqwamMatrixLibrary:createMatrix(#featureMatrix, numberOfClusters)
	
	for i = 1, #featureMatrix, 1 do

		local featureVector = {featureMatrix[i]}

		for j = 1, numberOfClusters, 1 do

			local weight = piMatrix[j][1]

			local meanVector = {meanMatrix[j]}

			local varianceVector = {varianceMatrix[j]}

			local probabilitiesVector = gaussian(featureVector, meanVector, varianceVector, epsilon)

			for i, probability in ipairs(probabilitiesVector[1]) do weight *= probability end

			probabilitiesMatrix[i][j] = weight

		end

	end
	
	return probabilitiesMatrix
	
end

function ExpectationMaximizationModel:initializeParameters(numberOfClusters, numberOfFeatures)

	local piMatrix = self:initializeMatrixBasedOnMode(numberOfClusters, 1)

	local meanMatrix = self:initializeMatrixBasedOnMode(numberOfClusters, numberOfFeatures)

	local varianceMatrix = self:initializeMatrixBasedOnMode(numberOfClusters, numberOfFeatures)

	return piMatrix, meanMatrix, varianceMatrix
	
end

local function eStep(featureMatrix, numberOfClusters, piMatrix, meanMatrix, varianceMatrix, epsilon)
	
	local numberOfData = #featureMatrix
	
	local responsibilitiesMatrix = calculateGaussianMatrix(featureMatrix, piMatrix, meanMatrix, varianceMatrix, epsilon) -- number of data x number of columns
	
	local responsibilitiesSumVector = AqwamMatrixLibrary:verticalSum(responsibilitiesMatrix)
	
	local normalizedResponsibilitiesMatrix = AqwamMatrixLibrary:divide(responsibilitiesMatrix, responsibilitiesSumVector)
	
	return normalizedResponsibilitiesMatrix
	
end

-- This function updates the model parameters based on the responsibility matrix

local function mStep(featureMatrix, responsibilitiesMatrix, numberOfClusters) -- data x features, data x clusters

	local numberOfData = #featureMatrix

	local numberOfFeatures = #featureMatrix[1]

	local piMatrix = AqwamMatrixLibrary:verticalSum(responsibilitiesMatrix)

	piMatrix = AqwamMatrixLibrary:divide(piMatrix, numberOfData)

	piMatrix = AqwamMatrixLibrary:transpose(piMatrix)

	local responsibilitiesMatrixTransposed = AqwamMatrixLibrary:transpose(responsibilitiesMatrix) -- clusters x data

	local sumWeight = AqwamMatrixLibrary:horizontalSum(responsibilitiesMatrixTransposed) -- clusters x 1

	local sumWeightX = AqwamMatrixLibrary:dotProduct(responsibilitiesMatrixTransposed, featureMatrix) -- clusters x features

	local meanMatrix = AqwamMatrixLibrary:divide(sumWeightX, sumWeight) -- clusters x features

	local varianceMatrix = AqwamMatrixLibrary:createMatrix(numberOfClusters, numberOfFeatures)

	for i = 1, numberOfClusters, 1 do

		local meanVector = {meanMatrix[i]}

		local thisStandardDeviationMatrix = AqwamMatrixLibrary:subtract(featureMatrix, meanVector)

		local thisVariationMatrix = AqwamMatrixLibrary:power(thisStandardDeviationMatrix, 2)

		local thisSumVariationMatrix = AqwamMatrixLibrary:verticalSum(thisVariationMatrix)

		varianceMatrix[i] = thisSumVariationMatrix[1]

	end

	varianceMatrix = AqwamMatrixLibrary:divide(varianceMatrix, sumWeight)

	return piMatrix, meanMatrix, varianceMatrix

end

function ExpectationMaximizationModel:getBayesianInformationCriterion(featureMatrix, numberOfClusters, epsilon)
	
	local numberOfFeatures = #featureMatrix[1]
	
	local piMatrix, meanMatrix, varianceMatrix = self:initializeParameters(numberOfClusters, numberOfFeatures)
	
	local responsibilities = eStep(featureMatrix, numberOfClusters, piMatrix, meanMatrix, varianceMatrix, epsilon)
	
	local piMatrix, meanMatrix, varianceMatrix = mStep(featureMatrix, responsibilities, numberOfClusters)
	
	local gaussianMatrix = calculateGaussianMatrix(featureMatrix, piMatrix, meanMatrix, varianceMatrix, epsilon)
	
	local likelihood = AqwamMatrixLibrary:applyFunction(math.log, gaussianMatrix)

	local sumLikelihood = AqwamMatrixLibrary:sum(likelihood)
	
	local numberOfData = #featureMatrix
	
	local numberOfFeatures = numberOfClusters * #featureMatrix[1]
	
	local bayesianInformationCriterion = (-2 * sumLikelihood) + (math.log(numberOfData) * numberOfFeatures)
	
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
	
	print(bestNumberOfClusters)

	return bestNumberOfClusters
	
end

function ExpectationMaximizationModel.new(maxNumberOfIterations, numberOfClusters, epsilon, targetCost)

	local NewExpectationMaximizationModel = BaseModel.new()

	setmetatable(NewExpectationMaximizationModel, ExpectationMaximizationModel)

	NewExpectationMaximizationModel.numberOfClusters = numberOfClusters or defaultNumberOfClusters

	NewExpectationMaximizationModel.epsilon = epsilon or defaultEpsilon

	NewExpectationMaximizationModel.maxNumberOfIterations = maxNumberOfIterations or defaultMaxNumberOfIterations

	NewExpectationMaximizationModel.targetCost = targetCost or defaultTargetCost

	return NewExpectationMaximizationModel
end

function ExpectationMaximizationModel:setParameters(maxNumberOfIterations, numberOfClusters, epsilon, targetCost)

	self.numberOfClusters = numberOfClusters or self.numberOfClusters

	self.epsilon = epsilon or self.epsilon

	self.maxNumberOfIterations = maxNumberOfIterations or self.maxNumberOfIterations

	self.targetCost = targetCost

end

function ExpectationMaximizationModel:train(featureMatrix)
	
	local piMatrix
	
	local meanMatrix
	
	local varianceMatrix
	
	local responsibilities
	
	local gaussianMatrix

	local costArray = {}
	
	local likelihoodArray = {}
	
	local cost = 1
	
	local numberOfIterations = 0

	local likelihood
	
	local sumLikelihood
	
	local numberOfFeatures = #featureMatrix[1]

	if (self.ModelParameters) then

		piMatrix, meanMatrix, varianceMatrix = table.unpack(self.ModelParameters)

		if (#featureMatrix[1] ~= #meanMatrix[1]) then error("The number of features are not the same as the model parameters!") end
		
	else
		
		if (self.numberOfClusters == math.huge) then
			
			self.numberOfClusters = self:fetchBestNumberOfClusters(featureMatrix, self.epsilon)
			
		end
		
		piMatrix, meanMatrix, varianceMatrix = self:initializeParameters(self.numberOfClusters, numberOfFeatures) 

	end

	repeat
		
		self:iterationWait()

		responsibilities = eStep(featureMatrix, self.numberOfClusters, piMatrix, meanMatrix, varianceMatrix, self.epsilon)

		piMatrix, meanMatrix, varianceMatrix = mStep(featureMatrix, responsibilities, self.numberOfClusters)
		
		gaussianMatrix = calculateGaussianMatrix(featureMatrix, piMatrix, meanMatrix, varianceMatrix, self.epsilon)
		
		likelihood = AqwamMatrixLibrary:applyFunction(math.log, gaussianMatrix)

		sumLikelihood = AqwamMatrixLibrary:sum(likelihood)
		
		table.insert(likelihoodArray, sumLikelihood)
		
		if (#likelihoodArray > 1) then
			
			cost = sumLikelihood - likelihoodArray[#likelihoodArray - 1] 
			
		end
		
		table.insert(costArray, cost)
		
		if (cost ~= cost) then error("Too much variance in the data! Please change the argument values.") end
		
		numberOfIterations += 1
		
		self:printCostAndNumberOfIterations(cost, numberOfIterations)

	until (numberOfIterations >= self.maxNumberOfIterations) or (cost <= self.targetCost)

	self.ModelParameters = {piMatrix, meanMatrix, varianceMatrix}

	return costArray

end

function ExpectationMaximizationModel:predict(featureMatrix, returnOriginalOutput)
	
	local piMatrix, meanMatrix, varianceMatrix = table.unpack(self.ModelParameters)

	local gaussianMatrix = calculateGaussianMatrix(featureMatrix, piMatrix, meanMatrix, varianceMatrix, self.epsilon)
	
	if (returnOriginalOutput == true) then return gaussianMatrix end
	
	local selectedClustersVector = AqwamMatrixLibrary:createMatrix(#featureMatrix, 1)

	local probabilityVector = AqwamMatrixLibrary:createMatrix(#featureMatrix, 1)
	
	for dataIndex, gausssianVector in ipairs(gaussianMatrix) do
		
		local selectedCluster
		
		local highestWeight = -math.huge
		
		for clusterNumber, weight in ipairs(gausssianVector) do
			
			if (weight < highestWeight) then continue end
				
			selectedCluster = clusterNumber
				
			highestWeight = weight
				
		end
		
		selectedClustersVector[dataIndex][1] = selectedCluster
		
		probabilityVector[dataIndex][1] = highestWeight
		
	end
	
	return selectedClustersVector, probabilityVector
	
end

return ExpectationMaximizationModel
