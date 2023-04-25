MachineLearningBaseModel = {}

MachineLearningBaseModel.__index = MachineLearningBaseModel

function MachineLearningBaseModel.new()
	
	local NewMachineLearningBaseModel = {}
	
	setmetatable(NewMachineLearningBaseModel, MachineLearningBaseModel)
	
	NewMachineLearningBaseModel.IsOutputPrinted = true

	NewMachineLearningBaseModel.ModelParameters = nil
	
	NewMachineLearningBaseModel.LastPredictedOutput = nil
	
	NewMachineLearningBaseModel.LastCalculations = nil

	return NewMachineLearningBaseModel
	
end

function MachineLearningBaseModel:getModelParameters()
	
	return self.ModelParameters
	
end

function MachineLearningBaseModel:setModelParameters(ModelParameters)
	
	self.ModelParameters = ModelParameters or self.ModelParameters
	
end

function MachineLearningBaseModel:clearModelParameters()
	
	self.ModelParameters = nil
	
end

function MachineLearningBaseModel:clearLastPredictedOutput()
	
	self.LastPredictedOutput = nil
	
end

function MachineLearningBaseModel:clearLastCalculations()
	
	self.LastCalculations = nil
	
end

function MachineLearningBaseModel:clearLastPredictedOutputAndCalculations()
	
	MachineLearningBaseModel:clearLastCalculations()
	
	MachineLearningBaseModel:clearLastPredictedOutput()
	
end

function MachineLearningBaseModel:printCostAndNumberOfIterations(cost, numberOfIteration) -- apparently it cannot see the self.isOutputPrinted when inherited function is used, so extra variable needed here
	
	 if self.IsOutputPrinted then print("Iteration: " .. numberOfIteration .. "\t\tCost: " .. cost) end

end


function MachineLearningBaseModel:setPrintOutput(option) 
	
	if (option == false) then
		
		self.IsOutputPrinted = false
		
	else
		
		self.IsOutputPrinted = true
		
	end
	
end

function MachineLearningBaseModel:getBooleanOrDefaultOption(boolean, defaultBoolean)
	
	if (boolean == true) then
		
		return true
		
	elseif (boolean == false) then
		
		return false
		
	else
		
		return defaultBoolean
		
	end
	
end


function MachineLearningBaseModel:destroy()
	
	setmetatable(self, nil)
	
	table.clear(self)
	
	table.freeze(self)
	
end

return MachineLearningBaseModel



