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