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

local IsotonicRegressionModel = {}

IsotonicRegressionModel.__index = IsotonicRegressionModel

setmetatable(IsotonicRegressionModel, IterativeMethodBaseModel)

local defaultIsIncreasing = true

local defaultOnOutOfBounds = "nan"

function IsotonicRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewIsotonicRegressionModel = IterativeMethodBaseModel.new(parameterDictionary)

	setmetatable(NewIsotonicRegressionModel, IsotonicRegressionModel)

	NewIsotonicRegressionModel:setName("IsotonicRegression")
	
	NewIsotonicRegressionModel.isIncreasing = NewIsotonicRegressionModel:getValueOrDefaultValue(parameterDictionary.isIncreasing, defaultIsIncreasing)
	
	NewIsotonicRegressionModel.onOutOfBounds = parameterDictionary.onOutOfBounds or defaultOnOutOfBounds

	return NewIsotonicRegressionModel
	
end

function IsotonicRegressionModel:train(featureMatrix, labelVector)
	
	local numberOfData = #featureMatrix

	if (numberOfData ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	if (#featureMatrix[1] ~= 1) then error("The feature matrix must only have 1 column.") end
	
	if (#labelVector[1] ~= 1) then error("The label matrix must only have 1 column.") end
	
	local isIncreasing = self.isIncreasing
	
	local sortConditionFunction = (isIncreasing and function(a, b) return a[1] < b[1] end) or function(a, b) return a[1] > b[1] end
	
	local sortedDataMatrix = {}
	
	for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do
		
		sortedDataMatrix[dataIndex] = {unwrappedFeatureVector[1], labelVector[dataIndex][1]}
		
	end
	
	table.sort(sortedDataMatrix, sortConditionFunction)
	
	local metaDataMatrix = {}
	
	local costArray = {}
	
	local numberOfInformation = numberOfData
	
	local numberOfIterations = 0
	
	local labelValue
	
	for dataIndex, unwrappedSortedDataVector in ipairs(sortedDataMatrix) do
		
		labelValue = unwrappedSortedDataVector[2]
		
		-- {startIndex, endIndex, totalWeight, totalValue, averageValue}
		
		metaDataMatrix[dataIndex] = {dataIndex, dataIndex, 1, labelValue, labelValue}
		
	end
	
	local isViolationFound
	
	local metaDataIndex
	
	local unwrappedCurrentMetaDataVector
	
	local unwrappedNextMetaDataVector
	
	local unwrappedMergedMetaDataVector
	
	local currentAverageValue
	
	local nextAverageValue 
	
	local isViolated
	
	local newTotalWeight
	
	local totalValue
	
	local averageValue
	
	local cost
		
	repeat
		
		numberOfIterations = numberOfIterations + 1
		
		isViolationFound = false
		
		metaDataIndex = 1

		while (metaDataIndex < numberOfInformation) do
			
			unwrappedCurrentMetaDataVector = metaDataMatrix[metaDataIndex]
			
			unwrappedNextMetaDataVector = metaDataMatrix[metaDataIndex + 1]
			
			currentAverageValue = unwrappedCurrentMetaDataVector[5]
			
			nextAverageValue = unwrappedNextMetaDataVector[5]
			
			isViolated = (isIncreasing and (currentAverageValue > nextAverageValue)) or ((not isIncreasing) and (currentAverageValue < nextAverageValue))

			if (isViolated) then
				
				isViolationFound = true
				
				newTotalWeight = unwrappedCurrentMetaDataVector[3] + unwrappedNextMetaDataVector[3]
				
				totalValue =  unwrappedCurrentMetaDataVector[4] + unwrappedNextMetaDataVector[4]
				
				averageValue = totalValue / newTotalWeight
				
				unwrappedMergedMetaDataVector = {unwrappedCurrentMetaDataVector[1], unwrappedNextMetaDataVector[2], newTotalWeight, totalValue, averageValue}
				
				metaDataMatrix[metaDataIndex] = unwrappedMergedMetaDataVector
				
				table.remove(metaDataMatrix, metaDataIndex + 1)

				numberOfInformation = numberOfInformation - 1
				
				cost = self:calculateCostWhenRequired(numberOfIterations, function()

					return math.pow(currentAverageValue - averageValue, 2) + math.pow(nextAverageValue - averageValue, 2)

				end)

				if (cost) then 

					table.insert(costArray, cost)

					self:printNumberOfIterationsAndCost(numberOfIterations, cost)

				end

				if (metaDataIndex > 1) then metaDataIndex = metaDataIndex - 1 end
				
			else
				
				metaDataIndex = metaDataIndex + 1
				
			end
			
		end
		
	until (not isViolationFound) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	local ModelParameters = {}
	
	local minimumFeatureValue

	local maximumFeatureValue

	local startIndex
	
	local endIndex
	
	for informationIndex, unwrappedInformationVector in ipairs(metaDataMatrix) do
		
		startIndex = unwrappedInformationVector[1]
		
		endIndex = unwrappedInformationVector[2]
		
		if (startIndex > endIndex) then startIndex, endIndex = endIndex, startIndex end
		
		minimumFeatureValue = sortedDataMatrix[startIndex][1]

		maximumFeatureValue = sortedDataMatrix[endIndex][1]
		
		if (minimumFeatureValue > maximumFeatureValue) then
			
			minimumFeatureValue, maximumFeatureValue = maximumFeatureValue, minimumFeatureValue
			
		end

		ModelParameters[informationIndex] = {minimumFeatureValue, maximumFeatureValue, unwrappedInformationVector[5]}
		
	end
	
	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	self.ModelParameters = ModelParameters
	
	return costArray

end

function IsotonicRegressionModel:predict(featureMatrix)
	
	if (#featureMatrix[1] ~= 1) then error("The feature matrix must only have 1 column.") end
	
	local onOutOfBounds = self.onOutOfBounds
	
	local ModelParameters = self.ModelParameters
	
	local numberOfData = #featureMatrix
	
	local nanValue = 0 / 0
	
	if (not ModelParameters) then return AqwamTensorLibrary:createTensor({numberOfData, 1}, nanValue) end
	
	local numberOfInformation = #ModelParameters
	
	local predictedLabelVector = {}
	
	local featureValue
	
	local predictedLabelValue
	
	local hasTargetValue
	
	local minimumFeatureValue
	
	local maximumFeatureValue
	
	local targetLabelValue
	
	for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do
		
		featureValue = unwrappedFeatureVector[1]
		
		predictedLabelValue = nil
		
		for informationIndex, unwrappedInformationVector in ipairs(ModelParameters) do
			
			minimumFeatureValue, maximumFeatureValue, targetLabelValue = table.unpack(unwrappedInformationVector)
			
			if (featureValue >= minimumFeatureValue) and (featureValue <= maximumFeatureValue) then

				predictedLabelValue = targetLabelValue
			
			elseif (featureValue < minimumFeatureValue) and (informationIndex == 1) then
				
				predictedLabelValue = ((onOutOfBounds == "clamp") and targetLabelValue) or nanValue
				
			elseif (featureValue > maximumFeatureValue) and (informationIndex == numberOfInformation) then
				
				predictedLabelValue = ((onOutOfBounds == "clamp") and targetLabelValue) or nanValue
				
			end
			
			if (predictedLabelValue) then break end
			
		end
		
		predictedLabelVector[dataIndex] = {predictedLabelValue}
		
	end
	
	return predictedLabelVector
	
end

return IsotonicRegressionModel
