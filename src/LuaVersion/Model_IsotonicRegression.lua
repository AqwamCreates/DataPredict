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

local IsotonicRegressionModel = {}

IsotonicRegressionModel.__index = IsotonicRegressionModel

setmetatable(IsotonicRegressionModel, IterativeMethodBaseModel)

local defaultIsIncreasing = true

local defaultMode = "Hybrid"

local defaultOnOutOfBounds = "NotANumber"

function IsotonicRegressionModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewIsotonicRegressionModel = IterativeMethodBaseModel.new(parameterDictionary)

	setmetatable(NewIsotonicRegressionModel, IsotonicRegressionModel)

	NewIsotonicRegressionModel:setName("IsotonicRegression")
	
	NewIsotonicRegressionModel.isIncreasing = NewIsotonicRegressionModel:getValueOrDefaultValue(parameterDictionary.isIncreasing, defaultIsIncreasing)
	
	NewIsotonicRegressionModel.mode = parameterDictionary.mode or defaultMode
	
	NewIsotonicRegressionModel.onOutOfBounds = parameterDictionary.onOutOfBounds or defaultOnOutOfBounds

	return NewIsotonicRegressionModel
	
end

function IsotonicRegressionModel:train(featureMatrix, labelVector)
	
	local numberOfData = #featureMatrix

	if (numberOfData ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	if (#featureMatrix[1] ~= 1) then error("The feature matrix must only have 1 column.") end
	
	if (#labelVector[1] ~= 1) then error("The label matrix must only have 1 column.") end
	
	local isIncreasing = self.isIncreasing
	
	local mode = self.mode
	
	local sortConditionFunction = (isIncreasing and function(a, b) return a[1] < b[1] end) or function(a, b) return a[1] > b[1] end
	
	local sortedDataMatrix = {}
	
	for dataIndex, unwrappedFeatureVector in ipairs(featureMatrix) do
		
		sortedDataMatrix[dataIndex] = {unwrappedFeatureVector[1], labelVector[dataIndex][1]}
		
	end
	
	table.sort(sortedDataMatrix, sortConditionFunction)
	
	local metaDataMatrix = {}
	
	local costArray = {}
	
	local numberOfInformation = numberOfData
	
	local numberOfIterations = 1
	
	local totalCost = 0
	
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
	
	local totalWeight
	
	local totalValue
	
	local averageValue
	
	local cost
		
	repeat
		
		isViolationFound = false
		
		metaDataIndex = 1
		
		cost = nil

		while (metaDataIndex < numberOfInformation) do
			
			unwrappedCurrentMetaDataVector = metaDataMatrix[metaDataIndex]
			
			unwrappedNextMetaDataVector = metaDataMatrix[metaDataIndex + 1]
			
			currentAverageValue = unwrappedCurrentMetaDataVector[5]
			
			nextAverageValue = unwrappedNextMetaDataVector[5]
			
			isViolated = (isIncreasing and (currentAverageValue > nextAverageValue)) or ((not isIncreasing) and (currentAverageValue < nextAverageValue))

			if (isViolated) then
				
				isViolationFound = true
				
				totalWeight = unwrappedCurrentMetaDataVector[3] + unwrappedNextMetaDataVector[3]
				
				totalValue =  unwrappedCurrentMetaDataVector[4] + unwrappedNextMetaDataVector[4]
				
				averageValue = totalValue / totalWeight
				
				unwrappedMergedMetaDataVector = {unwrappedCurrentMetaDataVector[1], unwrappedNextMetaDataVector[2], newTotalWeight, totalValue, averageValue}
				
				metaDataMatrix[metaDataIndex] = unwrappedMergedMetaDataVector
				
				table.remove(metaDataMatrix, metaDataIndex + 1)

				numberOfInformation = numberOfInformation - 1
				
				cost = self:calculateCostWhenRequired(numberOfIterations, function()

					return math.pow(currentAverageValue - averageValue, 2) + math.pow(nextAverageValue - averageValue, 2)

				end)

				if (cost) then totalCost = totalCost + cost end

				if (metaDataIndex > 1) then metaDataIndex = metaDataIndex - 1 end
				
			else
				
				metaDataIndex = metaDataIndex + 1
				
			end
			
		end
		
		if (cost) then
			
			table.insert(costArray, totalCost)

			self:printNumberOfIterationsAndCost(numberOfIterations, totalCost)
			
		end
		
		numberOfIterations = numberOfIterations + 1
		
	until (not isViolationFound) or self:checkIfTargetCostReached(cost) or self:checkIfConverged(cost)
	
	local informationMatrix = {}
	
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

		informationMatrix[informationIndex] = {minimumFeatureValue, maximumFeatureValue, unwrappedInformationVector[5]}
		
	end
	
	local ModelParameters = self.ModelParameters or {}
	
	local oldInformationMatrix = ModelParameters[1]
	
	local oldMetaDataMatrix = ModelParameters[2]
	
	local newMinimumFeatureValue
	
	local newMaximumFeatureValue
	
	local newTargetLabelValue
	
	local oldMinimumFeatureValue
	
	local oldMaximumFeatureValue
	
	local oldTargetLabelValue
	
	local mergedTargetLabelValue
	
	local mergedMinimumFeatureValue
	
	local mergedMaximumFeatureValue
	
	local unwrappedNewMetaDataVector
	
	local unwrappedOldMetaDataVector
	
	local newWeight
	
	local oldWeight
	
	local unwrappedCurrentMetaDataVector
	
	local unwrappedNextMetaDataVector
	
	local unwrappedCurrentInformationVector
	
	local unwrappedNextInformationVector
	
	local nextI
	
	if (mode == "Hybrid") then

		mode = (oldInformationMatrix and oldMetaDataMatrix and "Online") or "Offline"		

	end
	
	if (mode == "Online") then
		
		for newInformationIndex, unwrappedNewInformationVector in ipairs(informationMatrix) do

			for oldInformationIndex, unwrappedOldInformationVector in ipairs(oldInformationMatrix) do

				-- Check if the new information overlaps with old information.
				
				newMinimumFeatureValue, newMaximumFeatureValue, newTargetLabelValue = unwrappedNewInformationVector[1], unwrappedNewInformationVector[2], unwrappedNewInformationVector[3]
				
				oldMinimumFeatureValue, oldMaximumFeatureValue, oldTargetLabelValue = unwrappedOldInformationVector[1], unwrappedOldInformationVector[2], unwrappedOldInformationVector[3]

				-- Check for overlap between intervals.
				
				if (newMinimumFeatureValue <= oldMaximumFeatureValue and newMaximumFeatureValue >= oldMinimumFeatureValue) then
					
					-- Calculate merged interval and average.
					
					mergedMinimumFeatureValue = math.min(newMinimumFeatureValue, oldMinimumFeatureValue)
					
					mergedMaximumFeatureValue = math.max(newMaximumFeatureValue, oldMaximumFeatureValue)
					
					unwrappedNewMetaDataVector = metaDataMatrix[newInformationIndex]
					
					unwrappedOldMetaDataVector = oldMetaDataMatrix[oldInformationIndex]

					newWeight = (unwrappedNewMetaDataVector and unwrappedNewMetaDataVector[3]) or 1
					
					oldWeight = (unwrappedOldMetaDataVector and unwrappedOldMetaDataVector[3]) or 1

					mergedTargetLabelValue = (newTargetLabelValue * newWeight + oldTargetLabelValue * oldWeight) / (newWeight + oldWeight)

					-- Update the new information with merged values.
					
					unwrappedNewInformationVector[1] = mergedMinimumFeatureValue
					
					unwrappedNewInformationVector[2] = mergedMaximumFeatureValue
					
					unwrappedNewInformationVector[3] = mergedTargetLabelValue

					-- Update corresponding meta data.
					
					if (unwrappedNewMetaDataVector) then
						
						unwrappedNewMetaDataVector[3] = newWeight + oldWeight
						
						unwrappedNewMetaDataVector[4] = (newTargetLabelValue * newWeight) + (oldTargetLabelValue * oldWeight)
						
						unwrappedNewMetaDataVector[5] = mergedTargetLabelValue
						
					end
					
				end

			end

		end

		-- After merging, we need to ensure isotonic constraints are maintained.
		
		-- This would typically involve running the PAVA algorithm again or merging adjacent.
		
		-- intervals that violate the monotonic constraint.
		
		for i = 1, (#informationMatrix - 1) do
			
			unwrappedCurrentInformationVector = informationMatrix[i]
			
			unwrappedNextInformationVector = informationMatrix[i + 1]
			
			nextI = i + 1

			if (isIncreasing and (unwrappedCurrentInformationVector[3] > unwrappedNextInformationVector[3])) or ((not isIncreasing) and (unwrappedCurrentInformationVector[3] < unwrappedNextInformationVector[3])) then
				
				-- Merge violating intervals.
				
				mergedMinimumFeatureValue = math.min(unwrappedCurrentInformationVector[1], unwrappedNextInformationVector[1])
				
				mergedMaximumFeatureValue = math.max(unwrappedCurrentInformationVector[2], unwrappedNextInformationVector[2])

				unwrappedCurrentMetaDataVector = metaDataMatrix[i]
				
				unwrappedNextMetaDataVector = metaDataMatrix[nextI]

				totalWeight = unwrappedCurrentMetaDataVector[3] + unwrappedNextMetaDataVector[3]
				
				totalValue = unwrappedCurrentMetaDataVector[4] + unwrappedNextMetaDataVector[4]
				
				averageValue = totalValue / totalWeight

				unwrappedCurrentInformationVector[1] = mergedMinimumFeatureValue
				
				unwrappedCurrentInformationVector[2] = mergedMaximumFeatureValue
				
				unwrappedCurrentInformationVector[3] = averageValue

				unwrappedCurrentMetaDataVector[1] = math.min(unwrappedCurrentMetaDataVector[1], unwrappedNextMetaDataVector[1])  -- Start index.
				
				unwrappedCurrentMetaDataVector[2] = math.max(unwrappedCurrentMetaDataVector[2], unwrappedNextMetaDataVector[2])  -- End index.
				
				unwrappedCurrentMetaDataVector[3] = totalWeight
				
				unwrappedCurrentMetaDataVector[4] = totalValue
				
				unwrappedCurrentMetaDataVector[5] = averageValue

				table.remove(informationMatrix, nextI)
				
				table.remove(metaDataMatrix, nextI)

				if (i > 1) then i = i - 1 end
				
			end
			
		end
		
	end
	
	if (self.isOutputPrinted) then

		if (cost == math.huge) then warn("The model diverged.") end

		if (cost ~= cost) then warn("The model produced nan (not a number) values.") end

	end

	self.ModelParameters = {informationMatrix, metaDataMatrix}
	
	return costArray

end

function IsotonicRegressionModel:predict(featureMatrix)
	
	if (#featureMatrix[1] ~= 1) then error("The feature matrix must only have 1 column.") end
	
	local onOutOfBounds = self.onOutOfBounds
	
	local ModelParameters = self.ModelParameters or {}
	
	local informationMatrix = ModelParameters[1]
	
	local numberOfData = #featureMatrix
	
	local notANumberValue = 0 / 0
	
	if (not informationMatrix) then return AqwamTensorLibrary:createTensor({numberOfData, 1}, notANumberValue) end
	
	local numberOfInformation = #informationMatrix
	
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
		
		for informationIndex, unwrappedInformationVector in ipairs(informationMatrix) do
			
			minimumFeatureValue, maximumFeatureValue, targetLabelValue = table.unpack(unwrappedInformationVector)
			
			if (featureValue >= minimumFeatureValue) and (featureValue <= maximumFeatureValue) then

				predictedLabelValue = targetLabelValue
			
			elseif (featureValue < minimumFeatureValue) and (informationIndex == 1) then
				
				predictedLabelValue = ((onOutOfBounds == "Clamp") and targetLabelValue) or notANumberValue
				
			elseif (featureValue > maximumFeatureValue) and (informationIndex == numberOfInformation) then
				
				predictedLabelValue = ((onOutOfBounds == "Clamp") and targetLabelValue) or notANumberValue
				
			end
			
			if (predictedLabelValue) then break end
			
		end
		
		predictedLabelVector[dataIndex] = {predictedLabelValue}
		
	end
	
	return predictedLabelVector
	
end

return IsotonicRegressionModel
