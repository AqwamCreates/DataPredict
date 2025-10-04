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

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

local ModelTrainingModifier = require(script.Parent.ModelTrainingModifier)

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

local function getMaximumAcceptableCost(featureMatrix, labelMatrix, ClassesList)
	
	local absoluteFeatureMatrix = AqwamTensorLibrary:applyFunction(math.abs, featureMatrix)
	
	local sum = AqwamTensorLibrary:sum(absoluteFeatureMatrix)
	
	if (labelMatrix) and (ClassesList) then
		
		sum = sum + #labelMatrix
		
	elseif (labelMatrix) and (not ClassesList) then
		
		local absoluteLabelMatrix = AqwamTensorLibrary:applyFunction(math.abs, labelMatrix)

		sum = sum + AqwamTensorLibrary:sum(absoluteLabelMatrix)
		
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
		
		for _, value in ipairs(ModelParameters) do
			
			isAcceptable = checkIfModelParametersAreAcceptable(value)
			
			if (not isAcceptable) then return false end
			
		end
		
	else
		
		isAcceptable = checkIfIsAcceptableValue(ModelParameters, minimumValue, maximumValue)
		
	end
	
	return isAcceptable
	
end

local function removeDefectiveData(featureMatrix, labelMatrix, ClassesList) -- If even a single column contains a defective value, remove the whole row.
	
	local numberOfData = #featureMatrix

	local rowToDeleteArray = {}
	
	local numberOfClasses
	
	if (labelMatrix) then
		
		numberOfClasses = #labelMatrix[1]
		
	end
	
	for i, featureVector in ipairs(featureMatrix) do
		
		local isAcceptableData = true

		for f, featureValue in ipairs(featureVector) do

			if (not checkIfIsAcceptableValue(featureValue)) then
				
				rowToDeleteArray[i] = true
				
				isAcceptableData = false

				break
				
			end

		end
		
		if (labelMatrix) and (isAcceptableData) then
			
			local labelVector = labelMatrix[i]
			
			if (ClassesList) then
				
				if (not table.find(ClassesList, labelVector[1])) then
					
					rowToDeleteArray[i] = true

					isAcceptableData = false
					
					break
					
				end
				
			else
				
				for l = 1, numberOfClasses, 1 do

					if (not checkIfIsAcceptableValue(labelVector[l])) then

						rowToDeleteArray[i] = true

						isAcceptableData = false

						break

					end

				end
				
			end
			
		end
		
	end

	local filteredFeatureMatrix = {}
	
	local filteredLabelMatrix
	
	if (labelMatrix) then filteredLabelMatrix = {} end

	for i = 1, numberOfData, 1 do

		if (not rowToDeleteArray[i]) then

			table.insert(filteredFeatureMatrix, featureMatrix[i])

			if (labelMatrix) then table.insert(filteredLabelMatrix, labelMatrix[i]) end

		end

	end
	
	return filteredFeatureMatrix, filteredLabelMatrix

end

function ModelSafeguardWrapper.new(parameterDictionary)
	
	local NewModelSafeguardWrapper = BaseInstance.new(parameterDictionary)
	
	setmetatable(NewModelSafeguardWrapper, ModelSafeguardWrapper)
	
	NewModelSafeguardWrapper:setName("ModelSafeguardWrapper")
	
	NewModelSafeguardWrapper:setClassName("ModelSafeguardWrapper")
	
	local Model = parameterDictionary.Model
	
	NewModelSafeguardWrapper.Model = Model
	
	NewModelSafeguardWrapper.ignoreUpdateOnDefect = NewModelSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.ignoreUpdateOnDefect, defaultIgnoreUpdateOnDefect)
	
	NewModelSafeguardWrapper.removeDefectiveDataOnDefect = NewModelSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.removeDefectiveDataOnDefect, defaultRemoveDefectiveDataOnDefect)
	
	NewModelSafeguardWrapper.replaceValuesOnDefect = NewModelSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.replaceValuesOnDefect, defaultReplaceValuesOnDefect)
	
	NewModelSafeguardWrapper.modifyModelOnDefect = NewModelSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.modifyModelOnDefect, defaultModifyModelOnDefect)
	
	NewModelSafeguardWrapper.storeDefectiveUpdateInformation = NewModelSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.storeDefectiveUpdateInformation, defaultStoreDefectiveUpdateInformation)
	
	NewModelSafeguardWrapper.maximumAcceptableCostMultiplier = NewModelSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.maximumAcceptableCostMultiplier, defaultMaximumAcceptableCostMultiplier)
	
	NewModelSafeguardWrapper.ModifiedModel = parameterDictionary.ModifiedModel or ModelTrainingModifier.new({Model = Model, trainingMode = "Stochastic"})
	
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
	
	local isSuccessful = pcall(function()
		
		isAcceptable, valueArray = functionToRun()
		
	end)
	
	if (isSuccessful) and (isAcceptable) then
		
		self.canUseModel = true
		
		return table.unpack(valueArray or {})
		
	end
	
	if (storeDefectiveUpdateInformation) then

		local currentTimeString = tostring(os.time())

		defectiveUpdateInformationDictionary[currentTimeString] = eventName

	end
	
	Model:setModelParameters(OriginalModelParameters)

	if (ignoreUpdateOnDefect) or ((not ignoreUpdateOnDefect) and (not onDefectFunctionToRunDictionary)) then 

		self.canUseModel = true

		return table.unpack(valueArray or {}) 

	end
	
	local onDefectSettingArray = {self.removeDefectiveDataOnDefect,  self.replaceValuesOnDefect, self.modifyModelOnDefect}

	local onDefectFunctionNameArray = {"removeDefectFunction", "replaceValueFunction", "modifyModelFunction"}
	
	local onDefectFunctionName

	local onDefectFunctionToRun
	
	for i, value in ipairs(onDefectSettingArray) do
		
		if (value) then
			
			onDefectFunctionName = onDefectFunctionNameArray[i]
			
			onDefectFunctionToRun = onDefectFunctionToRunDictionary[onDefectFunctionName]
			
			if (onDefectFunctionToRun) then 
				
				onDefectFunctionToRun()
				
				Model:setModelParameters(OriginalModelParameters)
				
				isSuccessful = pcall(function()

					isAcceptable, valueArray = functionToRun()

				end)

				if (isSuccessful) and (isAcceptable) then

					self.canUseModel = true

					return table.unpack(valueArray or {})

				end
				
			end
			
		end
		
	end
	
	Model:setModelParameters(OriginalModelParameters)
	
	self.canUseModel = true
	
	return table.unpack(valueArray or {})
	
end

function ModelSafeguardWrapper:train(featureMatrix, labelMatrix)
	
	local Model = self.Model

	local ClassesList = Model.ClassesList
	
	local maximumAcceptableCostMultiplier = self.maximumAcceptableCostMultiplier
	
	local costArray
	
	local finalCostValue
	
	local isAcceptableValue
	
	local maximumAcceptableCost
	
	local onDefectFunctionToRunDictionary = {
		
		["removeDefectFunction"] = function()
			
			featureMatrix, labelMatrix = removeDefectiveData(featureMatrix, labelMatrix, ClassesList)
			
		end,
		
		["replaceValueFunction"] = function()
			
			if (not checkIfAllAreNumbers(featureMatrix)) then return end

			featureMatrix = AqwamTensorLibrary:zScoreNormalization(featureMatrix, 2)
			
		end,
		
		["modifyModelFunction"] = function()
			
			local ModifiedModel = self.ModifiedModel
			
			if (not ModifiedModel) then return end
			
			Model = ModifiedModel
			
		end,
		
	}
	
	self:runSandboxedEnvironment("train", Model, function()
		
		costArray = Model:train(featureMatrix, labelMatrix)
		
		maximumAcceptableCost = maximumAcceptableCostMultiplier * getMaximumAcceptableCost(featureMatrix, labelMatrix)

		finalCostValue = costArray[#costArray]
		
		isAcceptableValue = checkIfIsAcceptableValue(finalCostValue, -maximumAcceptableCost, maximumAcceptableCost)
		
		return isAcceptableValue, {costArray}
		
	end, onDefectFunctionToRunDictionary)
	
	return costArray or {}
	
end

function ModelSafeguardWrapper:update(...)
	
	local Model = self.Model
	
	local valueArray = {...}
	
	local UpdatedModelParameters

	self:runSandboxedEnvironment("update", Model, function()

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

function ModelSafeguardWrapper:getModelParameters(...)

	return self.Model:getModelParameters(...)

end

function ModelSafeguardWrapper:setModelParameters(...)

	self.Model:setModelParameters(...)

end

function ModelSafeguardWrapper:getCanUseModel()
	
	return self.canUseModel
	
end

return ModelSafeguardWrapper
