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

local IterativeMethodBaseModel = require("Model_IterativeMethodBaseModel")

GradientMethodBaseModel = {}

GradientMethodBaseModel.__index = GradientMethodBaseModel

setmetatable(GradientMethodBaseModel, IterativeMethodBaseModel)

function GradientMethodBaseModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewGradientMethodBaseModel = IterativeMethodBaseModel.new(parameterDictionary)
	
	setmetatable(NewGradientMethodBaseModel, GradientMethodBaseModel)
	
	NewGradientMethodBaseModel:setName("GradientMethodBaseModel")

	NewGradientMethodBaseModel:setClassName("GradientMethodModel")
	
	NewGradientMethodBaseModel.autoResetOptimizers = NewGradientMethodBaseModel:getValueOrDefaultValue(parameterDictionary.autoResetOptimizers, true)
	
	NewGradientMethodBaseModel.areGradientsSaved = NewGradientMethodBaseModel:getValueOrDefaultValue(parameterDictionary.areGradientsSaved, false)
	
	NewGradientMethodBaseModel.Gradients = NewGradientMethodBaseModel:getValueOrDefaultValue(parameterDictionary.Gradients, nil)
	
	return NewGradientMethodBaseModel
	
end

function GradientMethodBaseModel:setAutoResetOptimizers(option)

	self.autoResetOptimizers = self:getValueOrDefaultValue(option, self.autoResetOptimizers)

end

function GradientMethodBaseModel:setAreGradientsSaved(option)
	
	self.areGradientsSaved = self:getValueOrDefaultValue(option, self.areGradientsSaved)
	
end

function GradientMethodBaseModel:getGradients(doNotDeepCopy)
	
	if (doNotDeepCopy) then

		return self.Gradients

	else

		return self:deepCopyTable(self.Gradients)

	end
	
end

function GradientMethodBaseModel:setGradients(Gradients, doNotDeepCopy)
	
	if (doNotDeepCopy) then

		self.Gradients = Gradients

	else

		self.Gradients = self:deepCopyTable(Gradients)

	end

end

function GradientMethodBaseModel:clearGradients()
	
	self.Gradients = nil
	
end

return GradientMethodBaseModel