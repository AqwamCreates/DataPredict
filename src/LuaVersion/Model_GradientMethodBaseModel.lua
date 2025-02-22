--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local BaseModel = require("Model_BaseModel")

GradientMethodBaseModel = {}

GradientMethodBaseModel.__index = GradientMethodBaseModel

setmetatable(GradientMethodBaseModel, BaseModel)

function GradientMethodBaseModel.new()
	
	local NewGradientMethodBaseModel = BaseModel.new()
	
	setmetatable(NewGradientMethodBaseModel, GradientMethodBaseModel)
	
	NewGradientMethodBaseModel.autoResetOptimizers = true
	
	NewGradientMethodBaseModel.areGradientsSaved = false
	
	NewGradientMethodBaseModel.Gradients = nil
	
	return NewGradientMethodBaseModel
	
end

function BaseModel:setAutoResetOptimizers(option)

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
