BaseModel = {}

BaseModel.__index = BaseModel

function BaseModel.new()
	
	local NewBaseModel = {}
	
	setmetatable(NewBaseModel, BaseModel)
	
	NewBaseModel.IsOutputPrinted = true

	NewBaseModel.ModelParameters = nil
	
	NewBaseModel.LastPredictedOutput = nil
	
	NewBaseModel.LastCalculations = nil

	return NewBaseModel
	
end

function BaseModel:getModelParameters()
	
	return self.ModelParameters
	
end

function BaseModel:setModelParameters(ModelParameters)
	
	self.ModelParameters = ModelParameters or self.ModelParameters
	
end

function BaseModel:clearModelParameters()
	
	self.ModelParameters = nil
	
end

function BaseModel:clearLastPredictedOutput()
	
	self.LastPredictedOutput = nil
	
end

function BaseModel:clearLastCalculations()
	
	self.LastCalculations = nil
	
end

function BaseModel:clearLastPredictedOutputAndCalculations()
	
	BaseModel:clearLastCalculations()
	
	BaseModel:clearLastPredictedOutput()
	
end

function BaseModel:printCostAndNumberOfIterations(cost, numberOfIteration) -- apparently it cannot see the self.isOutputPrinted when inherited function is used, so extra variable needed here
	
	if self.IsOutputPrinted then print("Iteration: " .. numberOfIteration .. "\t\tCost: " .. cost) end

end


function BaseModel:setPrintOutput(option) 
	
	if (option == false) then
		
		self.IsOutputPrinted = false
		
	else
		
		self.IsOutputPrinted = true
		
	end
	
end

function BaseModel:getBooleanOrDefaultOption(boolean, defaultBoolean)
	
	if (boolean == true) then
		
		return true
		
	elseif (boolean == false) then
		
		return false
		
	else
		
		return defaultBoolean
		
	end
	
end


function BaseModel:destroy()
	
	setmetatable(self, nil)
	
	table.clear(self)
	
	table.freeze(self)
	
end

return BaseModel

