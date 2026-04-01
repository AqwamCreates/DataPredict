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

local TheilSenRegressionModel = {}

TheilSenRegressionModel.__index = TheilSenRegressionModel

setmetatable(TheilSenRegressionModel, BaseModel)

local function getMedianValueFromArray(array)
	
	local numberOfElements = #array
	
	local isNumberOfElementsEven = ((numberOfElements % 2) == 0)

	local chosenIndex

	local chosenValue
	
	table.sort(array)
	
	if (isNumberOfElementsEven) then

		local chosenIndex1 = (numberOfElements / 2)

		local chosenIndex2 = chosenIndex1 + 1

		local chosenValue1 = array[chosenIndex1]

		local chosenValue2 = array[chosenIndex2]

		chosenValue = (chosenValue1 + chosenValue2) / 2

	else

		chosenIndex = ((numberOfElements - 1) / 2) + 1

		chosenValue = array[chosenIndex]

	end
	
	return chosenValue
	
end

function TheilSenRegressionModel.new(parameterDictionary)

	local NewTheilSenRegressionModel = BaseModel.new(parameterDictionary)

	setmetatable(NewTheilSenRegressionModel, TheilSenRegressionModel)

	NewTheilSenRegressionModel:setName("TheilSenRegression")

	return NewTheilSenRegressionModel

end

function TheilSenRegressionModel:train(featureMatrix, labelVector)
	
	local numberOfData = #featureMatrix

	if (numberOfData ~= #labelVector) then error("The feature matrix and the label vector does not contain the same number of rows.") end
	
	if (#featureMatrix[1] ~= 1) then error("The feature matrix must only have 1 column.") end

	if (#labelVector[1] ~= 1) then error("The label matrix must only have 1 column.") end
	
	local slopeValueArray = {}
	
	local primaryFeatureValue
	
	local secondaryFeatureValue
	
	local primaryLabelValue
	
	local secondaryLabelValue
	
	local denominatorSlopeValue
	
	local nominatorSlopeValue
	
	local slopeValue
	
	for primaryIndex, unwrappedPrimaryFeatureVector in ipairs(featureMatrix) do
		
		primaryFeatureValue = unwrappedPrimaryFeatureVector[1]
		
		primaryLabelValue = labelVector[primaryIndex][1]
		
		for secondaryIndex = (primaryIndex + 1), numberOfData, 1 do
			
			secondaryFeatureValue = featureMatrix[secondaryIndex][1]

			secondaryLabelValue = labelVector[secondaryIndex][1]

			denominatorSlopeValue = secondaryFeatureValue - primaryFeatureValue

			if (denominatorSlopeValue ~= 0) then

				nominatorSlopeValue = secondaryLabelValue - primaryLabelValue

				slopeValue = nominatorSlopeValue / denominatorSlopeValue

				table.insert(slopeValueArray, slopeValue)

			end
			
		end
		
	end
	
	if (#slopeValueArray == 0) then error("All feature values are equal. No slope values can be used.") end

	local chosenSlopeValue = getMedianValueFromArray(slopeValueArray)
	
	local featureMatrixMultipliedBySlopeValue = AqwamTensorLibrary:multiply(featureMatrix, chosenSlopeValue)
	
	local medianBiasVector = AqwamTensorLibrary:subtract(labelVector, featureMatrixMultipliedBySlopeValue)
	
	local transposedMedianBiasVector = AqwamTensorLibrary:transpose(medianBiasVector)
	
	local medianBiasArray = transposedMedianBiasVector[1]
	
	local chosenBiasValue = getMedianValueFromArray(medianBiasArray)

	self.ModelParameters = {chosenSlopeValue, chosenBiasValue}

end

function TheilSenRegressionModel:predict(featureMatrix)
	
	local ModelParameters = self.ModelParameters or {}

	local medianSlopeValue = ModelParameters[1] or math.huge
	
	local medianBiasValue = ModelParameters[2] or math.huge
	
	local predictedLabelVector = AqwamTensorLibrary:multiply(featureMatrix, medianSlopeValue)
	
	predictedLabelVector = AqwamTensorLibrary:add(predictedLabelVector, medianBiasValue)

	return predictedLabelVector

end

return TheilSenRegressionModel
