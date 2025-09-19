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

local BaseIntstance = require(script.Parent.Parent.Cores.BaseInstance)

local ModelParametersSafeguardWrapper = {}

ModelParametersSafeguardWrapper.__index = ModelParametersSafeguardWrapper

setmetatable(ModelParametersSafeguardWrapper, BaseIntstance)

local defaultIgnoreUpdateOnDefect = false

local defaultRemoveDefectiveDataOnDefect = true

local defaultStoreDefectiveUpdateInformation = false

local function checkIfIsAcceptableValue(value)

	return (value == value) and (value ~= math.huge) and (value ~= -math.huge) and (type(value) == "number")

end

local function checkIfModelParametersAreAcceptable(ModelParameters)
	
	local isAcceptable = true
	
	if (type(ModelParameters) == "table") then
		
		for _, value in ModelParameters do
			
			isAcceptable = checkIfModelParametersAreAcceptable(ModelParameters)
			
			if (not isAcceptable) then return false end
			
		end
		
	else
		
		isAcceptable = checkIfIsAcceptableValue(ModelParameters)
		
	end
	
	return isAcceptable
	
end

function ModelParametersSafeguardWrapper.new(parameterDictionary)
	
	local NewModelParametersSafeguardWrapper = BaseIntstance.new(parameterDictionary)
	
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

function ModelParametersSafeguardWrapper:runSandboxedEnvironment(eventName, functionToRun)
	
	self.canUseModel = false

	local Model = self.Model

	local ignoreUpdateOnDefect = self.ignoreUpdateOnDefect

	local removeDefectiveDataOnDefect = self.removeDefectiveDataOnDefect

	local storeDefectiveUpdateInformation = self.storeDefectiveUpdateInformation
	
	local defectiveUpdateInformationDictionary = self.defectiveUpdateInformationDictionary

	local OriginalModelParameters = Model:getModelParameters()
	
	local isAcceptable = false
	
	local valueArray
	
	local currentTimeString
	
	repeat
		
		isAcceptable, valueArray = functionToRun(Model)

		if (not isAcceptable) then

			Model:setModelParameters(OriginalModelParameters)
			
			if (storeDefectiveUpdateInformation) then
				
				currentTimeString = tostring(os.time())
				
				defectiveUpdateInformationDictionary[currentTimeString] = eventName
				
			end

		end
		
	until (isAcceptable) or (ignoreUpdateOnDefect)
	
	self.canUseModel = true
	
	if (valueArray) then return table.unpack(valueArray) end
	
end

function ModelParametersSafeguardWrapper:train(...)
	
	local valueArray = {...}
	
	local costArray
	
	local finalCostValue
	
	local isAcceptableValue
	
	self:runSandboxedEnvironment("train", function(Model)
		
		costArray = Model:train(table.unpack(valueArray))

		finalCostValue = costArray[#costArray]
		
		isAcceptableValue = checkIfIsAcceptableValue(finalCostValue)
		
		return isAcceptableValue, {costArray}
		
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
