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

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

local ModelParametersSafeguardWrapper = {}

ModelParametersSafeguardWrapper.__index = ModelParametersSafeguardWrapper

setmetatable(ModelParametersSafeguardWrapper, BaseInstance)

local defaultIgnoreUpdateOnDefect = false

local defaultRemoveDefectiveDataOnDefect = true

local defaultStoreDefectiveUpdateInformation = false

local function checkIfIsAcceptableValue(value)

	return (value == value) and (value ~= math.huge) and (value ~= -math.huge)

end

local function checkIfModelParametersAreAcceptable(ModelParameters)
	
	local isAcceptable = true
	
	if (type(ModelParameters) == "table") then
		
		for _, value in ModelParameters do
			
			isAcceptable = checkIfModelParametersAreAcceptable(value)
			
			if (not isAcceptable) then return false end
			
		end
		
	else
		
		isAcceptable = checkIfIsAcceptableValue(ModelParameters)
		
	end
	
	return isAcceptable
	
end

local function removeDefectiveData(featureMatrix, labelMatrix) -- If even a single column contains a defective value, remove the whole row.
	
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

			for l = 1, numberOfClasses, 1 do

				if (not checkIfIsAcceptableValue(labelVector[l])) then

					rowToDeleteArray[i] = true

					isAcceptableData = false

					break

				end

			end
			
		end
		
	end

	local filteredFeatureMatrix = {}
	
	local filteredLabelMatrix = {}

	for i = 1, numberOfData, 1 do

		if (not rowToDeleteArray[i]) then
			
			table.insert(filteredFeatureMatrix, featureMatrix[i])
			
			if (labelMatrix) then table.insert(filteredLabelMatrix, labelMatrix[i]) end
			
		end

	end

	return filteredFeatureMatrix, filteredLabelMatrix

end

function ModelParametersSafeguardWrapper.new(parameterDictionary)
	
	local NewModelParametersSafeguardWrapper = BaseInstance.new(parameterDictionary)
	
	setmetatable(NewModelParametersSafeguardWrapper, ModelParametersSafeguardWrapper)
	
	NewModelParametersSafeguardWrapper:setName("ModelParametersSafeguardWrapper")
	
	NewModelParametersSafeguardWrapper:setClassName("ModelParametersSafeguardWrapper")
	
	NewModelParametersSafeguardWrapper.Model = parameterDictionary.Model
	
	NewModelParametersSafeguardWrapper.ignoreUpdateOnDefect = NewModelParametersSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.ignoreUpdateOnDefect, defaultIgnoreUpdateOnDefect)
	
	NewModelParametersSafeguardWrapper.removeDefectiveDataOnDefect = NewModelParametersSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.removeDefectiveDataOnDefect, defaultRemoveDefectiveDataOnDefect)
	
	NewModelParametersSafeguardWrapper.storeDefectiveUpdateInformation = NewModelParametersSafeguardWrapper:getValueOrDefaultValue(parameterDictionary.storeDefectiveUpdateInformation, defaultStoreDefectiveUpdateInformation)
	
	NewModelParametersSafeguardWrapper.canUseModel = true
	
	NewModelParametersSafeguardWrapper.defectiveUpdateInformationDictionary = {}
	
	return NewModelParametersSafeguardWrapper
	
end

function ModelParametersSafeguardWrapper:runSandboxedEnvironment(eventName, functionToRun, removeDefectFunction)
	
	self.canUseModel = false

	local Model = self.Model

	local ignoreUpdateOnDefect = self.ignoreUpdateOnDefect

	local removeDefectiveDataOnDefect = self.removeDefectiveDataOnDefect

	local storeDefectiveUpdateInformation = self.storeDefectiveUpdateInformation
	
	local defectiveUpdateInformationDictionary = self.defectiveUpdateInformationDictionary

	local OriginalModelParameters = Model:getModelParameters()
	
	local isAcceptable, valueArray = functionToRun(Model)
	
	if (isAcceptable) then
		
		self.canUseModel = true
		
		return table.unpack(valueArray or {})
		
	end
	
	if (storeDefectiveUpdateInformation) then

		local currentTimeString = tostring(os.time())

		defectiveUpdateInformationDictionary[currentTimeString] = eventName

	end
	
	Model:setModelParameters(OriginalModelParameters)

	if (ignoreUpdateOnDefect) then 

		self.canUseModel = true

		return table.unpack(valueArray or {}) 

	end

	if (removeDefectiveDataOnDefect) and (removeDefectFunction) then

		removeDefectFunction()

	end
	
	isAcceptable, valueArray = functionToRun(Model)
	
	return table.unpack(valueArray or {})
	
end

function ModelParametersSafeguardWrapper:train(featureMatrix, labelMatrix)
	
	local costArray
	
	local finalCostValue
	
	local isAcceptableValue
	
	self:runSandboxedEnvironment("train", function(Model)
		
		costArray = Model:train(featureMatrix, labelMatrix)

		finalCostValue = costArray[#costArray]
		
		isAcceptableValue = checkIfIsAcceptableValue(finalCostValue)
		
		return isAcceptableValue, {costArray}
		
	end, function()
		
		featureMatrix, labelMatrix = removeDefectiveData(featureMatrix, labelMatrix)
		
	end)
	
end

function ModelParametersSafeguardWrapper:update(...)
	
	local valueArray = {...}
	
	local UpdatedModelParameters

	self:runSandboxedEnvironment("update", function(Model)

		Model:update(table.unpack(valueArray))
		
		UpdatedModelParameters = Model:getModelParameters()

		return checkIfModelParametersAreAcceptable(UpdatedModelParameters)

	end)
	
end

function ModelParametersSafeguardWrapper:predict(...)
	
	return self.Model:predict(...)
	
end

function ModelParametersSafeguardWrapper:setModel(Model)
	
	self.Model = Model
	
end

function ModelParametersSafeguardWrapper:getModel()

	return self.Model

end

function ModelParametersSafeguardWrapper:getModelParameters(...)

	return self.Model:getModelParameters(...)

end

function ModelParametersSafeguardWrapper:setModelParameters(...)

	self.Model:setModelParameters(...)

end

function ModelParametersSafeguardWrapper:getCanUseModel()
	
	return self.canUseModel
	
end

return ModelParametersSafeguardWrapper
