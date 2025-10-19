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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local IterativeMethodBaseModel = require(script.Parent.IterativeMethodBaseModel)

local distanceFunctionDictionary = require(script.Parent.Parent.Cores.DistanceFunctionDictionary)

local AffinityPropagationModel = {}

AffinityPropagationModel.__index = AffinityPropagationModel

setmetatable(AffinityPropagationModel, IterativeMethodBaseModel)

local defaultMaxNumberOfIterations = 500

local defaultDamping = 0.5

local defaultDistanceFunction = "Euclidean"

local defaultPreferenceType = "Median"

local function createDistanceMatrix(distanceFunction, matrix1, matrix2)

	local numberOfData1 = #matrix1

	local numberOfData2 = #matrix2

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData1, numberOfData2})

	for i = 1, numberOfData1, 1 do

		for j = 1, numberOfData2, 1 do

			distanceMatrix[i][j] = distanceFunction({matrix1[i]}, {matrix2[j]})

		end

	end

	return distanceMatrix

end

local function getMedian(array)
	
	table.sort(array)
	
	local mid = math.floor(#array / 2)
	
	if ((#array % 2) == 0) then
		
		return ((array[mid] + array[mid + 1]) / 2)
		
	else
		
		return array[mid + 1]
		
	end
	
end

local function getAverage(array)
	
	local total = 0
	
	for _, value in ipairs(array) do
		
		total = total + value
		
	end
	
	return (total / #array)
	
end

local function setPreferencesToSimilarityMatrix(similarityMatrix, numberOfData, preferenceType, preferenceValueArray)
	
	local preferenceValue
	
	local triangularElementArray = {} -- Collect upper triangular non-diagonal elements

	for i = 1, numberOfData, 1 do
		
		for j = i + 1, numberOfData, 1 do
			
			table.insert(triangularElementArray, similarityMatrix[i][j])
			
		end
		
	end

	if (preferenceType == "Median") then
		
		preferenceValue = getMedian(triangularElementArray)
		
	elseif (preferenceType == "Average") then

		preferenceValue = getAverage(triangularElementArray)

	elseif (preferenceType == "Minimum") then

		preferenceValue = math.min(table.unpack(triangularElementArray))
		
	elseif (preferenceType == "Maximum") then

		preferenceValue = math.max(table.unpack(triangularElementArray))
		
	elseif (preferenceType == "Precomputed") then
		
		if (preferenceValueArray == nil) then error("No preference value array!") end
		
		if (#preferenceValueArray ~= numberOfData) then error("The length of the preference value array is not equal to number of data!") end
		
	else

		error("Invalid preference type!")

	end
	
	for i = 1, numberOfData do -- Fill diagonal with the computed preference value
		
		if (preferenceType == "Precomputed") then
			
			similarityMatrix[i][i] = preferenceValueArray[i]
			
		else
			
			similarityMatrix[i][i] = preferenceValue
			
		end
		
	end
	
	return similarityMatrix

end

local function calculateResponsibilityMatrix(responsibilityMatrix, availabilityMatrix, similarityMatrix)
	
	local numberOfData = #responsibilityMatrix

	for i = 1, numberOfData do

		for j = 1, numberOfData do

			local maxResponsibility = -math.huge

			for k = 1, numberOfData do

				if (k == j) then continue end

				maxResponsibility = math.max(maxResponsibility, similarityMatrix[i][k] + availabilityMatrix[k][j])

			end

			responsibilityMatrix[i][j] = similarityMatrix[i][j] - maxResponsibility

		end

	end
	
	return responsibilityMatrix

end

local function calculateAvailibilityMatrix(responsibilityMatrix, availibilityMatrix, damping)
	
	local updateFactor = 1 - damping
	
	local numberOfData = #responsibilityMatrix

	for i = 1, numberOfData, 1 do

		for j = 1, numberOfData, 1 do
			
			local availability
			
			local sumMaxAvailability = 0

			if (i ~= j) then
				
				for k = 1, numberOfData, 1 do

					if (k == i) and (k == j) then continue end

					local maxAvailability = math.max(0, responsibilityMatrix[k][j])
					
					sumMaxAvailability = sumMaxAvailability + maxAvailability

				end
				
				availability = (updateFactor * (responsibilityMatrix[j][j] + sumMaxAvailability)) + (damping * availibilityMatrix[i][j])
				
				availability = math.min(0, availability)
				
			else
				
				for k = 1, numberOfData, 1 do

					if (k == i) then continue end

					local maxAvailability = math.max(0, responsibilityMatrix[k][j])

					sumMaxAvailability = sumMaxAvailability + maxAvailability

				end
				
				availability = (updateFactor * sumMaxAvailability) + (damping * availibilityMatrix[i][j])

			end
			
			availibilityMatrix[i][j] = availability

		end

	end
	
	return availibilityMatrix

end

local function calculateCost(clusterNumberArray, responsibilityMatrix)

	local totalCost = 0

	for i = 1, #clusterNumberArray do

		totalCost += responsibilityMatrix[i][clusterNumberArray[i]]

	end

	return totalCost

end

local function assignClusters(responsibilityMatrix, availibilityMatrix)
	
	local calculatedValuesMatrix = AqwamTensorLibrary:add(responsibilityMatrix, availibilityMatrix)
	
	local clusterNumberArray = {}
	
	for i = 1, #calculatedValuesMatrix, 1 do
		
		local calculatedValuesVector = {calculatedValuesMatrix[i]}
		
		local clusterIndexArray = AqwamTensorLibrary:findMaximumValueDimensionIndexArray(calculatedValuesVector)

		if (clusterIndexArray == nil) then continue end

		local clusterNumber = clusterIndexArray[2]

		clusterNumberArray[i] = clusterNumber
		
	end

	return clusterNumberArray

end

function AffinityPropagationModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	parameterDictionary.maximumNumberOfIterations = parameterDictionary.maximumNumberOfIterations or defaultMaxNumberOfIterations

	local NewAffinityPropagationModel = IterativeMethodBaseModel.new(parameterDictionary)

	setmetatable(NewAffinityPropagationModel, AffinityPropagationModel)
	
	NewAffinityPropagationModel:setName("AffinityPropagation")

	NewAffinityPropagationModel.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction
	
	NewAffinityPropagationModel.preferenceType = parameterDictionary.preferenceType or defaultPreferenceType

	NewAffinityPropagationModel.damping = parameterDictionary.damping or defaultDamping
	
	NewAffinityPropagationModel.preferenceValueArray = parameterDictionary.preferenceValueArray

	return NewAffinityPropagationModel

end

function AffinityPropagationModel:train(featureMatrix)
	
	local damping = self.damping
	
	local ModelParameters = self.ModelParameters or {}
	
	local maximumNumberOfIterations = self.maximumNumberOfIterations
	
	local distanceFunctionToApply = distanceFunctionDictionary[self.distanceFunction]
	
	if (not distanceFunctionToApply) then error("Unknown distance function.") end
	
	local numberOfData = #featureMatrix
	
	local dimensionSizeArray = {numberOfData, numberOfData}
	
	local responsibilityMatrix = ModelParameters[3] or AqwamTensorLibrary:createTensor(dimensionSizeArray)

	local availabilityMatrix = ModelParameters[4] or AqwamTensorLibrary:createTensor(dimensionSizeArray)
	
	local distanceMatrix = createDistanceMatrix(distanceFunctionToApply, featureMatrix, featureMatrix)

	local similarityMatrix = AqwamTensorLibrary:multiply(-1, distanceMatrix)

	local numberOfIterations = 0

	local costArray = {}

	local clusterNumberArray

	local cost

	similarityMatrix = setPreferencesToSimilarityMatrix(similarityMatrix, numberOfData, self.preferenceType, self.preferenceValueArray)

	repeat
		
		numberOfIterations += 1
		
		self:iterationWait()

		responsibilityMatrix = calculateResponsibilityMatrix(responsibilityMatrix, availabilityMatrix, similarityMatrix)

		availabilityMatrix = calculateAvailibilityMatrix(responsibilityMatrix, availabilityMatrix, damping)
		
		clusterNumberArray = assignClusters(responsibilityMatrix, availabilityMatrix)
		
		cost = self:calculateCostWhenRequired(numberOfIterations, function()
			
			return calculateCost(clusterNumberArray, responsibilityMatrix)
			
		end) 
		
		if cost then
			
			table.insert(costArray, cost)

			self:printNumberOfIterationsAndCost(numberOfIterations, cost)
			
		end
		
	until (numberOfIterations >= maximumNumberOfIterations) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)

	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	self.ModelParameters = {featureMatrix, clusterNumberArray, responsibilityMatrix, availabilityMatrix}

	return costArray

end

function AffinityPropagationModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters
	
	local dimensionSizeArray = {#featureMatrix, 1}
	
	if (ModelParameters) then
		
		local placeholderClusterVector = AqwamTensorLibrary:createTensor(dimensionSizeArray, math.huge)
		
		local placeholderSimilarityVector = AqwamTensorLibrary:createTensor(dimensionSizeArray, math.huge)
		
		return placeholderClusterVector, placeholderSimilarityVector
		
	end
	
	local storedFeatureMatrix, clusterNumberArray = table.unpack(ModelParameters)

	local maximumSimilarityVector = AqwamTensorLibrary:createTensor(dimensionSizeArray)
	
	local predictedClusterVector = AqwamTensorLibrary:createTensor(dimensionSizeArray)
	
	local storedFeatureMatrix, clusterNumberArray = table.unpack()
	
	local distanceFunctionToApply = distanceFunctionDictionary[self.distanceFunction]
	
	local distanceMatrix = createDistanceMatrix(distanceFunctionToApply, featureMatrix, storedFeatureMatrix)
	
	for i, unwrappedDistanceVector in ipairs(distanceMatrix) do

		local index = AqwamTensorLibrary:findMinimumValueDimensionIndexArray({unwrappedDistanceVector})

		if (index) then
			
			local storedFeatureMatrixRowIndex = index[2]

			predictedClusterVector[i][1] = clusterNumberArray[storedFeatureMatrixRowIndex]

			maximumSimilarityVector[i][1] = unwrappedDistanceVector[storedFeatureMatrixRowIndex]
			
		end
		
	end
	
	return predictedClusterVector, maximumSimilarityVector

end

return AffinityPropagationModel
