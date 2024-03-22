local BaseModel = require(script.Parent.BaseModel)

GradientMethodBaseModel = {}

GradientMethodBaseModel.__index = GradientMethodBaseModel

setmetatable(GradientMethodBaseModel, BaseModel)

function GradientMethodBaseModel.new()
	
	local NewGradientMethodBaseModel = BaseModel.new()
	
	NewGradientMethodBaseModel.isGradientSaved = false
	
	NewGradientMethodBaseModel.Gradient = nil
	
	return NewGradientMethodBaseModel
	
end

function GradientMethodBaseModel:setIsGradientSaved(option)
	
	self.isGradientSaved = self:getBooleanOrDefaultOption(option, self.isGradientSaved)
	
end

function GradientMethodBaseModel:getGradient(doNotDeepCopy)
	
	if doNotDeepCopy then

		return self.Gradient

	else

		return self:deepCopyTable(self.Gradient)

	end
	
end

function GradientMethodBaseModel:setGradient(gradient, doNotDeepCopy)

	if doNotDeepCopy then

		self.Gradient = gradient

	else

		self.Gradient = self:deepCopyTable(gradient)

	end

end

function GradientMethodBaseModel:clearGradient()
	
	self.Gradient = nil
	
end

return GradientMethodBaseModel
