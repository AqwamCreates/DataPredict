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

local ModelParametersSafeGuardWrapper = {}

ModelParametersSafeGuardWrapper.__index = ModelParametersSafeGuardWrapper

setmetatable(ModelParametersSafeGuardWrapper, BaseIntstance)

local function checkIfIsAcceptableValue(value)

	return (value == value) and (value ~= math.huge) and (value ~= -math.huge) and (type(value) == "number")

end

function ModelParametersSafeGuardWrapper.new(parameterDictionary)
	
	local NewModelParametersSafeGuardWrapper = BaseIntstance.new(parameterDictionary)
	
	setmetatable(NewModelParametersSafeGuardWrapper, ModelParametersSafeGuardWrapper)
	
	NewModelParametersSafeGuardWrapper:setName("ModelParametersSafeGuardWrapper")
	
	NewModelParametersSafeGuardWrapper:setClassName("ModelParametersSafeGuardWrapper")
	
	NewModelParametersSafeGuardWrapper.Model = parameterDictionary.Model
	
	NewModelParametersSafeGuardWrapper.canUseModel = true
	
	return NewModelParametersSafeGuardWrapper
	
end

function ModelParametersSafeGuardWrapper:train(...)
	
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

function ModelParametersSafeGuardWrapper:predict(...)
	
	return self.Model:predict(...)
	
end

function ModelParametersSafeGuardWrapper:update(...)
	
	self.canUseModel = false

	local Model = self.Model

	local OriginalModelParameters = Model:getModelParameters()
	
	while true do
		
		Model:update(...)
		
	end

end

function ModelParametersSafeGuardWrapper:setModel(Model)
	
	self.Model = Model
	
end

function ModelParametersSafeGuardWrapper:getModel()

	return self.Model

end

function ModelParametersSafeGuardWrapper:getCanUseModel()
	
	return self.canUseModel
	
end

return ModelParametersSafeGuardWrapper
