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

local function checkIfIsAcceptableValue(value)

	return (value == value) and (value ~= math.huge) and (value ~= -math.huge) and (type(value) == "number")

end

function ModelParametersSafeguardWrapper.new(parameterDictionary)
	
	local NewModelParametersSafeguardWrapper = BaseIntstance.new(parameterDictionary)
	
	setmetatable(NewModelParametersSafeguardWrapper, ModelParametersSafeguardWrapper)
	
	NewModelParametersSafeguardWrapper:setName("ModelParametersSafeguardWrapper")
	
	NewModelParametersSafeguardWrapper:setClassName("ModelParametersSafeguardWrapper")
	
	NewModelParametersSafeguardWrapper.Model = parameterDictionary.Model
	
	NewModelParametersSafeguardWrapper.canUseModel = true
	
	return NewModelParametersSafeguardWrapper
	
end

function ModelParametersSafeguardWrapper:train(...)
	
	self.canUseModel = false
	
	local Model = self.Model
	
	local OriginalModelParameters = Model:getModelParameters()
	
	local costArray
	
	while true do
		
		costArray = Model:train(...)
		
		local finalCostValue = costArray[#costArray]
		
		if (checkIfIsAcceptableValue(finalCostValue)) then

			self.canUseModel = true

			break

		end
		
		Model:setModelParameters(OriginalModelParameters)
		
	end
	
	return costArray
	
end

function ModelParametersSafeguardWrapper:update(...)

	self.canUseModel = false

	local Model = self.Model

	local OriginalModelParameters = Model:getModelParameters()

	while true do

		Model:update(...)

	end

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
