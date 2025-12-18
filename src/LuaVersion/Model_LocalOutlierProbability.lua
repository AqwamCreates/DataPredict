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

local BaseModel = require("Model_BaseModel")

local Cores = script.Parent.Parent.Cores

local distanceFunctionDictionary = require("Core_DistanceFunctionDictionary")

local ZTableFunction = require("Core_ZTableFunction")

local LocalOutlierProbability = {}

LocalOutlierProbability.__index = LocalOutlierProbability

setmetatable(LocalOutlierProbability, BaseModel)

local defaultKValue = 3

local defaultDistanceFunction = "Euclidean"

local defaultLambda = 3

local defaultMaximumNumberOfData = math.huge

local function moreThanOrEqualToZeroClampFunction(value)
	
	-- Slightly modified from original max(0, erf(value)) due to numerical issues arising from floating-point round-off when calculating decimals.
	
	return math.clamp(value, 0, 1) 
	
end

local function createDistanceMatrix(distanceFunction, featureMatrix)

	local numberOfData = #featureMatrix

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfData}, 0)

	local calculateDistance = distanceFunctionDictionary[distanceFunction]

	for i, primaryUnwrappedFeatureVector in ipairs(featureMatrix) do

		for j, secondaryUnwrappedFeatureVector in ipairs(featureMatrix) do

			distanceMatrix[i][j] = calculateDistance({primaryUnwrappedFeatureVector}, {secondaryUnwrappedFeatureVector})

		end

	end

	return distanceMatrix

end

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else -- number, string, boolean, etc

		copy = original

	end

	return copy

end

local function merge(unwrappedDistanceVector, left, mid, right)

	local subArrayOne = mid - left + 1
	local subArrayTwo = right - mid

	local unwrappedLeftDistanceVector = {}
	local unwrappedRightDistanceVector = {}

	for i = 1, subArrayOne do

		unwrappedLeftDistanceVector[i] = unwrappedDistanceVector[left + i - 1]

	end

	for j = 1, subArrayTwo do

		unwrappedRightDistanceVector[j] = unwrappedDistanceVector[mid + j]

	end

	local indexOfSubArrayOne = 1
	local indexOfSubArrayTwo = 1
	local indexOfMergedArray = left

	while indexOfSubArrayOne <= subArrayOne and indexOfSubArrayTwo <= subArrayTwo do

		if unwrappedLeftDistanceVector[indexOfSubArrayOne] <= unwrappedRightDistanceVector[indexOfSubArrayTwo] then

			unwrappedDistanceVector[indexOfMergedArray] = unwrappedLeftDistanceVector[indexOfSubArrayOne]
			indexOfSubArrayOne = indexOfSubArrayOne + 1

		else

			unwrappedDistanceVector[indexOfMergedArray] = unwrappedRightDistanceVector[indexOfSubArrayTwo]
			indexOfSubArrayTwo = indexOfSubArrayTwo + 1

		end

		indexOfMergedArray = indexOfMergedArray + 1

	end

	while (indexOfSubArrayOne <= subArrayOne) do

		unwrappedDistanceVector[indexOfMergedArray] = unwrappedLeftDistanceVector[indexOfSubArrayOne]
		indexOfSubArrayOne = indexOfSubArrayOne + 1
		indexOfMergedArray = indexOfMergedArray + 1

	end

	while (indexOfSubArrayTwo <= subArrayTwo) do

		unwrappedDistanceVector[indexOfMergedArray] = unwrappedRightDistanceVector[indexOfSubArrayTwo]
		indexOfSubArrayTwo = indexOfSubArrayTwo + 1
		indexOfMergedArray = indexOfMergedArray + 1

	end

end

local function mergeSort(distanceVector, startingValue, endValue)

	if (startingValue >= endValue) then return end

	local mid = math.floor(startingValue + (endValue - startingValue) / 2)

	mergeSort(distanceVector, startingValue, mid)
	mergeSort(distanceVector, mid + 1, endValue)
	merge(distanceVector, startingValue, mid, endValue)

end

function LocalOutlierProbability.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewLocalOutlierProbability = BaseModel.new(parameterDictionary)

	setmetatable(NewLocalOutlierProbability, LocalOutlierProbability)
	
	NewLocalOutlierProbability:setName("LocalOutlierProbability")

	NewLocalOutlierProbability.kValue = parameterDictionary.kValue or defaultKValue

	NewLocalOutlierProbability.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction
	
	NewLocalOutlierProbability.lamda = parameterDictionary.lamda or defaultLambda
	
	NewLocalOutlierProbability.maximumNumberOfData = parameterDictionary.maximumNumberOfData or defaultMaximumNumberOfData
	
	return NewLocalOutlierProbability

end

function LocalOutlierProbability:train(featureMatrix)

	local numberOfData = #featureMatrix

	local maximumNumberOfData = self.maximumNumberOfData

	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		local storedFeatureMatrix = ModelParameters[1]

		local storedLabelVector = ModelParameters[2]

		featureMatrix = AqwamTensorLibrary:concatenate(featureMatrix, storedFeatureMatrix, 1)

		numberOfData = #featureMatrix

		if (numberOfData > maximumNumberOfData) then

			local newFeatureMatrix = {}

			local newLabelVector = {}

			local dataShiftIndex = (numberOfData - maximumNumberOfData)

			for dataIndex = 1, maximumNumberOfData, 1 do

				newFeatureMatrix[dataIndex] = featureMatrix[dataIndex + dataShiftIndex]

			end

			featureMatrix = newFeatureMatrix

			numberOfData = maximumNumberOfData

		end

	end

	if (numberOfData <= self.kValue) and (self.isOutputPrinted) then warn("Number of data is less than or equal to the K value. Please add more data before doing any predictions.") end

	self.ModelParameters = featureMatrix

end

function LocalOutlierProbability:score()
	
	local featureMatrix = self.ModelParameters

	if (not featureMatrix) then return {{0}} end

	local kValue = self.kValue

	local distanceFunction = self.distanceFunction
	
	local lambda = self.lamda
	
	local numberOfData = #featureMatrix
	
	local numberOfDataMinusOne = numberOfData - 1
	
	local distanceMatrix = createDistanceMatrix(distanceFunction, featureMatrix)
	
	local nearestNeighbourIndexArrayArray = {}
	
	local numberOfNearestNeighboursVector = {}
	
	local sumDistanceVector = {}

	for i, unwrappedDistanceVector in ipairs(distanceMatrix) do

		local sortedUnwrappedDistanceVector = deepCopyTable(unwrappedDistanceVector)
		
		-- Ignores the distance to itself. It is removed before sorting to reduce computational resources.
		
		table.remove(sortedUnwrappedDistanceVector, i)

		mergeSort(sortedUnwrappedDistanceVector, 1, numberOfDataMinusOne)
		
		local kDistance = sortedUnwrappedDistanceVector[kValue]
		
		local nearestNeighbourIndexArray = {}
		
		local sumDistance = 0
		
		local numberOfNearestNeighbours = 0
		
		for j, distance in ipairs(unwrappedDistanceVector) do
			
			if (distance <= kDistance) then
				
				sumDistance = sumDistance + math.pow(distance, 2)
				
				table.insert(nearestNeighbourIndexArray, j) 
				
			end
			
		end
		
		sumDistanceVector[i] = {sumDistance}
		
		nearestNeighbourIndexArrayArray[i] = nearestNeighbourIndexArray
		
		numberOfNearestNeighboursVector[i] = {#nearestNeighbourIndexArray}

	end
	
	local standardDistanceVectorPart1 = AqwamTensorLibrary:divide(sumDistanceVector, numberOfNearestNeighboursVector)
	
	local standardDistanceVector = AqwamTensorLibrary:applyFunction(math.sqrt, standardDistanceVectorPart1)
	
	local probabilisticDistanceVector = AqwamTensorLibrary:multiply(lambda, standardDistanceVector)
	
	local probabilisticLocalOutlierProbabilityFactorVector = {}
	
	for i, nearestNeighboursIndexArray in ipairs(nearestNeighbourIndexArrayArray) do

		local meanProbabilityDistance = 0

		for _, nearestNeighbourIndex in ipairs(nearestNeighboursIndexArray) do

			meanProbabilityDistance = meanProbabilityDistance + probabilisticDistanceVector[nearestNeighbourIndex][1]

		end
		
		meanProbabilityDistance = meanProbabilityDistance / numberOfNearestNeighboursVector[i][1]
		
		local probabilisticLocalOutlierProbabilityFactor = (probabilisticDistanceVector[i][1] / meanProbabilityDistance) - 1
		
		probabilisticLocalOutlierProbabilityFactorVector[i] = {probabilisticLocalOutlierProbabilityFactor}

	end
	
	local nProbabilisticLocalOutlierProbabilityFactorVectorPart1 = AqwamTensorLibrary:power(probabilisticLocalOutlierProbabilityFactorVector, 2)
	
	local nProbabilisticLocalOutlierProbabilityFactorVectorPart2 = AqwamTensorLibrary:mean(nProbabilisticLocalOutlierProbabilityFactorVectorPart1)
	
	local nProbabilisticLocalOutlierProbabilityFactor = lambda * math.sqrt(nProbabilisticLocalOutlierProbabilityFactorVectorPart2)
	
	local gaussianErrorVectorPart2 = AqwamTensorLibrary:divide(probabilisticLocalOutlierProbabilityFactorVector, nProbabilisticLocalOutlierProbabilityFactor)
	
	local gaussianErrorVector = {}
	
	for i, unwrappedErrorVectorPart2 in ipairs(gaussianErrorVectorPart2) do
		
		gaussianErrorVector[i] = {ZTableFunction:getStandardNormalCumulativeDistributionFunctionValue(unwrappedErrorVectorPart2[1])}
		
	end
	
	local localOutlierProbabilityVector = AqwamTensorLibrary:applyFunction(moreThanOrEqualToZeroClampFunction, gaussianErrorVector)
	
	return localOutlierProbabilityVector

end

return LocalOutlierProbability
