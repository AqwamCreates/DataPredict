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

local BaseModel = require(script.Parent.BaseModel)

local distanceFunctionDictionary = require(script.Parent.Parent.Cores.DistanceFunctionDictionary)

local KNearestNeighboursClassifierModel = {}

KNearestNeighboursClassifierModel.__index = KNearestNeighboursClassifierModel

setmetatable(KNearestNeighboursClassifierModel, BaseModel)

local defaultKValue = 3

local defaultDistanceFunction = "Euclidean"

local defaultUseWeightedDistance = false

local defaultMaximumNumberOfData = math.huge

local function createDistanceMatrix(distanceFunction, featureMatrix, storedFeatureMatrix)

	local numberOfData = #featureMatrix

	local numberOfStoredData = #storedFeatureMatrix

	local distanceMatrix = AqwamTensorLibrary:createTensor({numberOfData, numberOfStoredData}, 0)

	local calculateDistance = distanceFunctionDictionary[distanceFunction]

	for datasetIndex = 1, numberOfData, 1 do

		for storedDatasetIndex = 1, numberOfStoredData, 1 do

			distanceMatrix[datasetIndex][storedDatasetIndex] = calculateDistance({featureMatrix[datasetIndex]}, {storedFeatureMatrix[storedDatasetIndex]})

		end

	end

	return distanceMatrix

end

local function merge(unwrappedDistanceVector, labelVector, left, mid, right)

	local subArrayOne = mid - left + 1
	local subArrayTwo = right - mid

	local unwrappedLeftDistanceVector = {}
	local unwrappedRightDistanceVector = {}

	local unwrappedLeftLabelVector = {}
	local unwrappedRightLabelVector = {}

	for i = 1, subArrayOne do

		unwrappedLeftDistanceVector[i] = unwrappedDistanceVector[left + i - 1]
		unwrappedLeftLabelVector[i] = labelVector[left + i - 1][1]

	end

	for j = 1, subArrayTwo do

		unwrappedRightDistanceVector[j] = unwrappedDistanceVector[mid + j]
		unwrappedRightLabelVector[j] = labelVector[mid + j][1]

	end

	local indexOfSubArrayOne = 1
	local indexOfSubArrayTwo = 1
	local indexOfMergedArray = left

	while (indexOfSubArrayOne <= subArrayOne) and (indexOfSubArrayTwo <= subArrayTwo) do

		if (unwrappedLeftDistanceVector[indexOfSubArrayOne] <= unwrappedRightDistanceVector[indexOfSubArrayTwo]) then

			unwrappedDistanceVector[indexOfMergedArray] = unwrappedLeftDistanceVector[indexOfSubArrayOne]
			labelVector[indexOfMergedArray][1] = unwrappedLeftLabelVector[indexOfSubArrayOne]
			indexOfSubArrayOne = indexOfSubArrayOne + 1

		else

			unwrappedDistanceVector[indexOfMergedArray] = unwrappedRightDistanceVector[indexOfSubArrayTwo]
			labelVector[indexOfMergedArray][1] = unwrappedRightLabelVector[indexOfSubArrayTwo]
			indexOfSubArrayTwo = indexOfSubArrayTwo + 1

		end

		indexOfMergedArray = indexOfMergedArray + 1

	end

	while (indexOfSubArrayOne <= subArrayOne) do

		unwrappedDistanceVector[indexOfMergedArray] = unwrappedLeftDistanceVector[indexOfSubArrayOne]
		labelVector[indexOfMergedArray][1] = unwrappedLeftLabelVector[indexOfSubArrayOne]
		indexOfSubArrayOne = indexOfSubArrayOne + 1
		indexOfMergedArray = indexOfMergedArray + 1

	end

	while (indexOfSubArrayTwo <= subArrayTwo) do

		unwrappedDistanceVector[indexOfMergedArray] = unwrappedRightDistanceVector[indexOfSubArrayTwo]
		labelVector[indexOfMergedArray][1] = unwrappedRightLabelVector[indexOfSubArrayTwo]
		indexOfSubArrayTwo = indexOfSubArrayTwo + 1
		indexOfMergedArray = indexOfMergedArray + 1

	end

end

local function mergeSort(unwrappedDistanceVector, labelVector, startingValue, endValue)

	if (startingValue >= endValue) then return end

	local mid = math.floor(startingValue + (endValue - startingValue) / 2)

	mergeSort(unwrappedDistanceVector, labelVector, startingValue, mid)
	mergeSort(unwrappedDistanceVector, labelVector, mid + 1, endValue)
	merge(unwrappedDistanceVector, labelVector, startingValue, mid, endValue)

end

local function getMajorityClass(sortedLabelVectorLowestToHighest, unwrappedDistanceVector, kValue, useWeightedDistance)

	local classWeights = {}

	local minimumNumberOfkValue = math.min(#sortedLabelVectorLowestToHighest, kValue)

	for k = 1, minimumNumberOfkValue, 1 do

		local label = sortedLabelVectorLowestToHighest[k][1]

		local distance = unwrappedDistanceVector[k]

		local weight

		if (useWeightedDistance) then

			weight = ((distance == 0) and math.huge) or (1 / distance)

		else

			weight = 1

		end

		classWeights[label] = (classWeights[label] or 0) + weight

	end

	local majorityClass, maxWeight = nil, -math.huge

	for label, weight in pairs(classWeights) do

		if weight > maxWeight then

			majorityClass = label

			maxWeight = weight

		end

	end

	return majorityClass

end

function KNearestNeighboursClassifierModel.new(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local NewKNearestNeighboursClassifierModel = BaseModel.new(parameterDictionary)

	setmetatable(NewKNearestNeighboursClassifierModel, KNearestNeighboursClassifierModel)
	
	NewKNearestNeighboursClassifierModel:setName("KNearestNeighboursClassifier")

	NewKNearestNeighboursClassifierModel.kValue = parameterDictionary.kValue or defaultKValue

	NewKNearestNeighboursClassifierModel.distanceFunction = parameterDictionary.distanceFunction or defaultDistanceFunction

	NewKNearestNeighboursClassifierModel.useWeightedDistance = NewKNearestNeighboursClassifierModel:getValueOrDefaultValue(parameterDictionary.useWeightedDistance, defaultUseWeightedDistance)
	
	NewKNearestNeighboursClassifierModel.maximumNumberOfData = parameterDictionary.maximumNumberOfData or defaultMaximumNumberOfData
	
	return NewKNearestNeighboursClassifierModel

end

function KNearestNeighboursClassifierModel:train(featureMatrix, labelVector)
	
	local numberOfData = #featureMatrix

	if (numberOfData ~= #labelVector) then error("The number of data in the feature matrix and the label vector are not the same.") end
	
	local maximumNumberOfData = self.maximumNumberOfData

	local ModelParameters = self.ModelParameters

	if (ModelParameters) then

		local storedFeatureMatrix = ModelParameters[1]

		local storedLabelVector = ModelParameters[2]

		featureMatrix = AqwamTensorLibrary:concatenate(featureMatrix, storedFeatureMatrix, 1)

		labelVector = AqwamTensorLibrary:concatenate(labelVector, storedLabelVector, 1)
		
		numberOfData = #featureMatrix
		
		if (numberOfData > maximumNumberOfData) then
			
			local newFeatureMatrix = {}
			
			local newLabelVector = {}
			
			local dataShiftIndex = (numberOfData - maximumNumberOfData)
			
			for dataIndex = 1, maximumNumberOfData, 1 do
				
				newFeatureMatrix[dataIndex] = featureMatrix[dataIndex + dataShiftIndex]
				
				newLabelVector[dataIndex] = labelVector[dataIndex + dataShiftIndex]
				
			end
			
			featureMatrix = newFeatureMatrix
			
			labelVector = newLabelVector
			
			numberOfData = maximumNumberOfData
			
		end

	end

	if (numberOfData < self.kValue) and (self.isOutputPrinted) then warn("Number of data is less than the K value. Please add more data before doing any predictions.") end

	self.ModelParameters = {featureMatrix, labelVector}

end

function KNearestNeighboursClassifierModel:predict(featureMatrix, returnOriginalOutput)

	local ModelParameters = self.ModelParameters

	if (not ModelParameters) then 

		local unknownValue = (returnOriginalOutput and math.huge) or nil

		return AqwamTensorLibrary:createTensor({#featureMatrix, 1}, unknownValue) 

	end

	local storedFeatureMatrix = ModelParameters[1]

	local storedLabelVector = ModelParameters[2]

	local kValue = self.kValue

	local distanceFunction = self.distanceFunction

	local useWeightedDistance = self.useWeightedDistance

	local distanceMatrix = createDistanceMatrix(distanceFunction, featureMatrix, storedFeatureMatrix)

	if (returnOriginalOutput) then return distanceMatrix end
	
	local numberOfOtherData = #storedFeatureMatrix

	local predictedLabelVector = {}
	
	for i, unwrappedDistanceVector in ipairs(distanceMatrix) do
		
		local sortedUnwrappedDistanceVectorLowestToHighest = self:deepCopyTable(unwrappedDistanceVector)

		local sortedLabelVectorLowestToHighest = self:deepCopyTable(storedLabelVector)

		mergeSort(sortedUnwrappedDistanceVectorLowestToHighest, sortedLabelVectorLowestToHighest, 1, numberOfOtherData)

		local majorityClass = getMajorityClass(sortedLabelVectorLowestToHighest, sortedUnwrappedDistanceVectorLowestToHighest, kValue, useWeightedDistance)

		predictedLabelVector[i] = {majorityClass}
		
	end

	return predictedLabelVector

end

return KNearestNeighboursClassifierModel
