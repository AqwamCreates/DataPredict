local BaseModel = require(script.Parent.BaseModel)

GradientMethodBaseModel = {}

GradientMethodBaseModel.__index = GradientMethodBaseModel

setmetatable(GradientMethodBaseModel, BaseModel)

function GradientMethodBaseModel.new()
	
	local NewGradientMethodBaseModel = BaseModel.new()
	
	NewGradientMethodBaseModel.areGradientsSaved = false
	
	NewGradientMethodBaseModel.Gradients = nil
	
	return NewGradientMethodBaseModel
	
end

function GradientMethodBaseModel:setAreGradientsSaved(option)
	
	self.areGradientsSaved = self:getBooleanOrDefaultOption(option, self.areGradientsSaved)
	
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
