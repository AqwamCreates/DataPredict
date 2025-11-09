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

local BaseInstance = require("Core_BaseInstance")

local ModelTrainingModifier = require("Other_ModelTrainingModifier")

local ModelSafeguardWrapper = {}

ModelSafeguardWrapper.__index = ModelSafeguardWrapper

setmetatable(ModelSafeguardWrapper, BaseInstance)

local defaultIgnoreUpdateOnDefect = false

local defaultRemoveDefectiveDataOnDefect = true

local defaultReplaceValuesOnDefect = true

local defaultModifyModelOnDefect = true

local defaultStoreDefectiveUpdateInformation = false

local defaultMaximumAcceptableCostMultiplier = 1

local function checkIfAllAreNumbers(matrix)
	
	local numberOfRows = #matrix
	
	local numberOfColumns = #matrix[1]
	
	for row = 1, numberOfRows, 1 do
		
		for column = 1, numberOfColumns, 1 do
			
			if (type(matrix[row][column]) ~= "number") then return false end
			
		end
		
	end
	
	return true
	
end

local function getMaximumAcceptableCost(dataMatrixArray, hasClassification)
	
	local sum = 0
	
	local partialSum
	
	local absoluteDataMatrix
	
	for i, dataMatrix in ipairs(dataMatrixArray) do
		
		if (i == 2) and (hasClassification) and (#dataMatrix[1] == 1) then
			
			partialSum = #dataMatrix
			
		else
			
			absoluteDataMatrix = AqwamTensorLibrary:applyFunction(math.abs, dataMatrix)
			
			partialSum = AqwamTensorLibrary:sum(absoluteDataMatrix)
			
		end
		
		sum = sum + partialSum
		
	end
	
	return sum
	
end

local function checkIfIsAcceptableValue(value, minimumValue, maximumValue)
	
	local isValidValue = (value == value) and (value ~= math.huge) and (value ~= -math.huge) and (type(value) == "number")
	
	if (not isValidValue) then return false end
	
	if (minimumValue) then
		
		if (value < minimumValue) then return false end
		
	end
	
	if (maximumValue) then

		if (value > maximumValue) then return false end

	end

	return true

end

local function checkIfModelParametersAreAcceptable(ModelParameters, minimumValue, maximumValue)
	
	local isAcceptable = true
	
	if (type(ModelParameters) == "table") then
		
		for _, value in pairs(ModelParameters) do
			
			isAcceptable = checkIfModelParametersAreAcceptable(value, minimumValue, maximumValue)
			
			if (not isAcceptable) then return false end
			
		end
		
	else
		
		isAcceptable = checkIfIsAcceptableValue(ModelParameters, minimumValue, maximumValue)
		
	end
	
	return isAcceptable
	
end

local function markRowsWithUnknownClass(dataMatrix, ClassesList)

	local numberOfData = #dataMatrix

	local rowWithUnknownClassArray = {}

	local index = 1

	for i, unwrappedDataVector in ipairs(dataMatrix) do

		if (not table.find(ClassesList, unwrappedDataVector[1])) then

			rowWithUnknownClassArray[index] = i

			index = index + 1

		end

	end

	return rowWithUnknownClassArray

end

local function markRowsWithDefectiveData(dataMatrix)

	local numberOfData = #dataMatrix

	local rowWithDefectiveDataArray = {}

	local index = 1

	for i, unwrappedDataVector in ipairs(dataMatrix) do

		for f, value in ipairs(unwrappedDataVector) do

			if (not checkIfIsAcceptableValue(value)) then

				rowWithDefectiveDataArray[index] = i

				index = index + 1

				break

			end

		end

	end

	return rowWithDefectiveDataArray

end

local function mergeRowWithDefectiveDataArrays(rowWithDefectiveDataArrayArray)
	
	local mergedArray = {}
	
	local seenDictionary = {}
	
	local index = 1

	for _, array in ipairs(rowWithDefectiveDataArrayArray) do
		
		for _, item in ipairs(array) do
			
			if (not seenDictionary[item]) then
				
				mergedArray[index] = item
				
				seenDictionary[item] = true
				
				index = index + 1
				
			end
			
		end
		
	end

	return mergedArray
	
end

local function removeRows(rowToRemoveArray, dataMatrix)
	
	local newDataMatrix = {}
	
	local index = 1
	
	for i, unwrappedDataVector in ipairs(dataMatrix) do
		
		if (not table.find(rowToRemoveArray, i)) then
			
			newDataMatrix[index] = unwrappedDataVector
			
			index = index + 1
			
		end
		
	end
	
	return newDataMatrix
	
end

-- If even a single column contains a defective value, remove the whole row.

local function removeDefectiveData(dataMatrixArray, hasClassification, ClassesList)
	
	local rowWithDefectiveDataArrayArray = {}
	
	local newDataMatrixArray = {}
	
	for i, dataMatrix in ipairs(dataMatrixArray) do
		
		if (i == 2) and (hasClassification) and (#dataMatrix[1] == 1) then
			
			rowWithDefectiveDataArrayArray[i] = markRowsWithUnknownClass(dataMatrix, ClassesList)
			
		else
			
			rowWithDefectiveDataArrayArray[i] = markRowsWithDefectiveData(dataMatrix)
			
		end
		
	end
	
	local rowWithDefectiveDataArray = mergeRowWithDefectiveDataArrays(rowWithDefectiveDataArrayArray)
	
	for i, dataMatrix in ipairs(dataMatrixArray) do
		
		newDataMatrixArray[i] = removeRows(rowWithDefectiveDataArray, dataMatrix)
		
	end
	
	return newDataMatrixArray

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

function ModelSafeguardWrapper.new(parameterDictionary)
	
	local NewModelSafeguardWrapper = BaseInstance.new(parameterDictionary)
	
	setmetatable(NewModelSafeguardWrapper, ModelSafeguardWrapper)
	
	NewModelSafeguardWrapper:setName("ModelSafeguardWrapper")
	
	NewModelSafeguardWrapper:setClassName("ModelSafeguardWrapper")
	
	local Model = parameterDictionary.Model
	
	local isOutputPrinted = NewModelSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.isOutputPrinted, Model.isOutputPrinted)
	
	NewModelSafeguardWrapper.Model = Model
	
	NewModelSafeguardWrapper.ignoreUpdateOnDefect = NewModelSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.ignoreUpdateOnDefect, defaultIgnoreUpdateOnDefect)
	
	NewModelSafeguardWrapper.removeDefectiveDataOnDefect = NewModelSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.removeDefectiveDataOnDefect, defaultRemoveDefectiveDataOnDefect)
	
	NewModelSafeguardWrapper.replaceValuesOnDefect = NewModelSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.replaceValuesOnDefect, defaultReplaceValuesOnDefect)
	
	NewModelSafeguardWrapper.modifyModelOnDefect = NewModelSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.modifyModelOnDefect, defaultModifyModelOnDefect)
	
	NewModelSafeguardWrapper.storeDefectiveUpdateInformation = NewModelSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.storeDefectiveUpdateInformation, defaultStoreDefectiveUpdateInformation)
	
	NewModelSafeguardWrapper.maximumAcceptableCostMultiplier = NewModelSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.maximumAcceptableCostMultiplier, defaultMaximumAcceptableCostMultiplier)
	
	NewModelSafeguardWrapper.ModifiedModel = parameterDictionary.ModifiedModel or ModelTrainingModifier.new({Model = Model, mode = "Stochastic", isOutputPrinted = isOutputPrinted})
	
	NewModelSafeguardWrapper.canUseModel = true
	
	NewModelSafeguardWrapper.defectiveUpdateInformationDictionary = {}
	
	return NewModelSafeguardWrapper
	
end

function ModelSafeguardWrapper:runSandboxedEnvironment(eventName, Model, functionToRun, onDefectFunctionToRunDictionary)
	
	self.canUseModel = false

	local ignoreUpdateOnDefect = self.ignoreUpdateOnDefect

	local storeDefectiveUpdateInformation = self.storeDefectiveUpdateInformation
	
	local defectiveUpdateInformationDictionary = self.defectiveUpdateInformationDictionary

	local OriginalModelParameters = Model:getModelParameters()
	
	local isAcceptable = false
	
	local valueArray
	
	local currentTimeString
	
	self.OriginalModelParameters = OriginalModelParameters
	
	local isSuccessful = pcall(function()
		
		isAcceptable, valueArray = functionToRun()
		
	end)
	
	if (isSuccessful) and (isAcceptable) then
		
		self.canUseModel = true
		
		self.OriginalModelParameters = nil
		
		return table.unpack(valueArray or {})
		
	end
	
	if (storeDefectiveUpdateInformation) then

		defectiveUpdateInformationDictionary[tostring(os.time())] = eventName

	end
	
	Model:setModelParameters(OriginalModelParameters)

	if (ignoreUpdateOnDefect) or ((not ignoreUpdateOnDefect) and (not onDefectFunctionToRunDictionary)) then 

		self.canUseModel = true
		
		self.OriginalModelParameters = nil

		return table.unpack(valueArray or {})

	end
	
	local onDefectSettingArray = {self.removeDefectiveDataOnDefect,  self.replaceValuesOnDefect, self.removeDefectiveDataOnDefect, self.modifyModelOnDefect}

	local onDefectFunctionNameArray = {"removeDefectFunction", "replaceValueFunction", "removeDefectFunction", "modifyModelFunction"}
	
	local onDefectFunctionName

	local onDefectFunctionToRun
	
	local canFixDefect
	
	for i, value in ipairs(onDefectSettingArray) do
		
		if (value) then
			
			onDefectFunctionName = onDefectFunctionNameArray[i]
			
			onDefectFunctionToRun = onDefectFunctionToRunDictionary[onDefectFunctionName]
			
			if (onDefectFunctionToRun) then 
				
				canFixDefect = pcall(onDefectFunctionToRun)
				
				Model:setModelParameters(OriginalModelParameters)
				
				if (not canFixDefect) then
					
					if (storeDefectiveUpdateInformation) then

						defectiveUpdateInformationDictionary[tostring(os.time())] = eventName .. " + " .. onDefectFunctionName

					end
					
					break 
					
				end
				
				isSuccessful = pcall(function()

					isAcceptable, valueArray = functionToRun()

				end)

				if (isSuccessful) and (isAcceptable) then

					self.canUseModel = true
					
					self.OriginalModelParameters = nil

					return table.unpack(valueArray or {})

				end
				
			end
			
		end
		
	end
	
	Model:setModelParameters(OriginalModelParameters)
	
	self.canUseModel = true
	
	self.OriginalModelParameters = nil
	
	return table.unpack(valueArray or {})
	
end

function ModelSafeguardWrapper:train(...)
	
	local Model = self.Model

	local ClassesList = Model.ClassesList or Model.StatesList or Model.ObservationStatesList
	
	local maximumAcceptableCostMultiplier = self.maximumAcceptableCostMultiplier
	
	local isTable = (type(ClassesList) == "table")
	
	local numberOfClasses = (isTable and #ClassesList) or 0
	
	local hasClassification = (numberOfClasses ~= 0)
	
	local dataMatrixArray = {...}
	
	local numberOfDataMatrix = #dataMatrixArray
	
	local numberOfData
	
	local costArray
	
	local finalCostValue
	
	local maximumAcceptableCost
	
	local isAcceptableValue
	
	local UpdatedModelParameters
	
	local valueToReturnArray
	
	local onDefectFunctionToRunDictionary = {
		
		["removeDefectFunction"] = function()
			
			dataMatrixArray = removeDefectiveData(dataMatrixArray, hasClassification)
			
		end,
		
		["replaceValueFunction"] = function()
			
			for i, dataMatrix in ipairs(dataMatrixArray) do
				
				if (i ~= 2) or (not hasClassification) or (#dataMatrix[1] ~= 1) then

					if (checkIfAllAreNumbers(dataMatrix)) then

						dataMatrixArray[i] = AqwamTensorLibrary:zScoreNormalization(dataMatrix, 2)

					end

				end
				
			end
			
		end,
		
		["modifyModelFunction"] = function()
			
			local ModifiedModel = self.ModifiedModel
			
			if (not ModifiedModel) then return end
			
			Model = ModifiedModel
			
		end,
		
	}
	
	return self:runSandboxedEnvironment("train", Model, function()
		
		valueToReturnArray = {}
		
		numberOfData = #dataMatrixArray[1]
		
		for i = 2, numberOfDataMatrix, 1 do
			
			if (numberOfData ~= #dataMatrixArray[i]) then error("The number of data for all input matrices are not equal.") end
			
		end
		
		-- No data means no training.
		
		if (numberOfData == 0) then return true, valueToReturnArray end
		
		costArray = Model:train(table.unpack(dataMatrixArray))
		
		UpdatedModelParameters = Model:getModelParameters()
		
		if (not costArray) then
			
			isAcceptableValue = checkIfModelParametersAreAcceptable(UpdatedModelParameters)
			
			return isAcceptableValue, valueToReturnArray
			
		end
		
		finalCostValue = costArray[#costArray]
		
		if (type(finalCostValue) == "nil") then

			isAcceptableValue = checkIfModelParametersAreAcceptable(UpdatedModelParameters)

			return isAcceptableValue, valueToReturnArray
			
		end
		
		maximumAcceptableCost = maximumAcceptableCostMultiplier * getMaximumAcceptableCost(dataMatrixArray, hasClassification)
		
		isAcceptableValue = checkIfIsAcceptableValue(finalCostValue, -maximumAcceptableCost, maximumAcceptableCost)
		
		valueToReturnArray[1] = costArray
		
		return isAcceptableValue, valueToReturnArray
		
	end, onDefectFunctionToRunDictionary)
	
end

function ModelSafeguardWrapper:update(...)
	
	local Model = self.Model
	
	local valueArray = {...}
	
	local UpdatedModelParameters

	return self:runSandboxedEnvironment("update", Model, function()

		Model:update(table.unpack(valueArray))
		
		UpdatedModelParameters = Model:getModelParameters()

		return checkIfModelParametersAreAcceptable(UpdatedModelParameters)

	end)
	
end

function ModelSafeguardWrapper:predict(...)
	
	return self.Model:predict(...)
	
end

function ModelSafeguardWrapper:setModel(Model)
	
	self.Model = Model
	
end

function ModelSafeguardWrapper:getModel()

	return self.Model

end

function ModelSafeguardWrapper:getModelParameters(doNotDeepCopy, ...)
	
	if (self.CanUseModel) then
		
		return self.Model:getModelParameters(doNotDeepCopy, ...)
		
	end
	
	local OriginalModelParameters = self.OriginalModelParameters
		
	if (doNotDeepCopy) then
		
		return OriginalModelParameters
		
	end
	
	return deepCopyTable(OriginalModelParameters)

end

function ModelSafeguardWrapper:setModelParameters(...)

	self.Model:setModelParameters(...)

end

function ModelSafeguardWrapper:getCanUseModel()
	
	return self.canUseModel
	
end

return ModelSafeguardWrapper
