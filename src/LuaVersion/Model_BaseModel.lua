--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

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

local BaseInstance = require("Core_BaseInstance")

local AqwamTensorLibrary = require("AqwamTensorLibrary")

BaseModel = {}

BaseModel.__index = BaseModel

setmetatable(BaseModel, BaseInstance)

function BaseModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseModel = BaseInstance.new(parameterDictionary)

	setmetatable(NewBaseModel, BaseModel)

	NewBaseModel:setName("BaseModel")

	NewBaseModel:setClassName("Model")
	
	NewBaseModel.isOutputPrinted = NewBaseModel:getValueOrDefaultValue(parameterDictionary.isOutputPrinted, true)

	NewBaseModel.ModelParameters = NewBaseModel:getValueOrDefaultValue(parameterDictionary.ModelParameters, nil)
	
	NewBaseModel.modelParametersInitializationMode = NewBaseModel:getValueOrDefaultValue(parameterDictionary.modelParametersInitializationMode, "RandomUniformNegativeAndPositive") 

	NewBaseModel.maximumModelParametersInitializationValue = NewBaseModel:getValueOrDefaultValue(parameterDictionary.maximumModelParametersInitializationValue, nil)
	
	NewBaseModel.minimumModelParametersInitializationValue = NewBaseModel:getValueOrDefaultValue(parameterDictionary.minimumModelParametersInitializationValue, nil)

	return NewBaseModel
	
end

function BaseModel:getModelParameters(doNotDeepCopy)
	
	if doNotDeepCopy then
		
		return self.ModelParameters
		
	else
		
		return self:deepCopyTable(self.ModelParameters)
		
	end
	
end

function BaseModel:setModelParameters(ModelParameters, doNotDeepCopy)
	
	if ModelParameters and doNotDeepCopy then
		
		self.ModelParameters = ModelParameters
		
	elseif ModelParameters and not doNotDeepCopy then
		
		self.ModelParameters = self:deepCopyTable(ModelParameters) 
		
	end
	
end

function BaseModel:clearModelParameters()
	
	self.ModelParameters = nil
	
end

function BaseModel:setPrintOutput(option) 
	
	self.isOutputPrinted = self:getValueOrDefaultValue(option, self.isOutputPrinted)
	
end

function BaseModel:setModelParametersInitializationMode(initializationMode, minimumModelParametersInitializationValue, maximumModelParametersInitializationValue)
	
	self.modelParametersInitializationMode = initializationMode
	
	self.minimumModelParametersInitializationValue = minimumModelParametersInitializationValue
	
	self.maximumModelParametersInitializationValue = maximumModelParametersInitializationValue
	
end

function BaseModel:initializeMatrixBasedOnMode(dimensionSizeArray, dimensionSizeToIgnoreArray) -- Some of the row/column might not be considered as an input variables/neurons. Hence, it should be ignored by subtracting from original rows and columns with the number of non-input variables/neurons.
	
	if (not dimensionSizeArray) then error("No dimension size array for weight initialization!") end
	
	dimensionSizeToIgnoreArray = dimensionSizeToIgnoreArray or {}

	local numberOfRowsToIgnore = dimensionSizeToIgnoreArray[1] or 0
	
	local numberOfColumnsToIgnore = dimensionSizeToIgnoreArray[2] or 0
	
	local adjustedNumberOfRows = dimensionSizeArray[1] - numberOfRowsToIgnore
	
	local adjustedNumberOfColumns = dimensionSizeArray[2] - numberOfColumnsToIgnore
	
	local numberOfDimensions = #dimensionSizeArray
	
	local initializationMode = self.modelParametersInitializationMode

	if (initializationMode == "Zero") then

		return AqwamTensorLibrary:createTensor(dimensionSizeArray, 0)

	elseif (initializationMode == "Random") then

		return AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray, self.minimumModelParametersInitializationValue, self.maximumModelParametersInitializationValue)

	elseif (initializationMode == "RandomNormal") then

		return AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray, self.modelParametersMean, self.modelParametersStandardDeviation)

	elseif (initializationMode == "RandomUniformPositive") then

		return AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

	elseif (initializationMode == "RandomUniformNegative") then

		local randomUniformTensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

		return AqwamTensorLibrary:multiply(randomUniformTensor, -1)

	elseif (initializationMode == "RandomUniformNegativeAndPositive") then

		local randomUniformTensor1 = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

		local randomUniformTensor2 = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

		return AqwamTensorLibrary:subtract(randomUniformTensor1, randomUniformTensor2)

	elseif (initializationMode == "HeNormal") then

		local variancePart1 = 2 / adjustedNumberOfRows

		local variancePart = math.sqrt(variancePart1)

		local randomNormalTensor = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)

		return AqwamTensorLibrary:multiply(variancePart, randomNormalTensor)

	elseif (initializationMode == "HeUniform") then

		local variancePart1 = 6 / adjustedNumberOfRows

		local variancePart = math.sqrt(variancePart1)

		local randomUniformTensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

		return AqwamTensorLibrary:multiply(variancePart, randomUniformTensor) 

	elseif (initializationMode == "XavierNormal") then

		local variancePart1 = 2 / (adjustedNumberOfRows + adjustedNumberOfColumns)

		local variancePart = math.sqrt(variancePart1)

		local randomNormalTensor = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)

		return AqwamTensorLibrary:multiply(variancePart, randomNormalTensor) 

	elseif (initializationMode == "XavierUniform") then

		local variancePart1 = 6 / (adjustedNumberOfRows + adjustedNumberOfColumns)

		local variancePart = math.sqrt(variancePart1)

		local randomUniformTensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

		return AqwamTensorLibrary:multiply(variancePart, randomUniformTensor)

	elseif (initializationMode == "LeCunNormal") then

		local variancePart1 = 1 / adjustedNumberOfRows

		local variancePart = math.sqrt(variancePart1)

		local randomNormalTensor = AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray)

		return AqwamTensorLibrary:multiply(variancePart, randomNormalTensor) 

	elseif (initializationMode == "LeCunUniform") then

		local variancePart1 = 3 / adjustedNumberOfRows

		local variancePart = math.sqrt(variancePart1)

		local randomUniformTensor = AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray)

		return AqwamTensorLibrary:multiply(variancePart, randomUniformTensor) 

	elseif (initializationMode == "None") then

		return nil

	else

		error("Invalid weight initialization mode.")

	end
	
end

return BaseModel