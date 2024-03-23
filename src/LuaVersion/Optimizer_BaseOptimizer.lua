BaseOptimizer = {}

BaseOptimizer.__index = BaseOptimizer

function BaseOptimizer.new(optimizerName)
	
	local NewBaseOptimizer = {}
	
	setmetatable(NewBaseOptimizer, BaseOptimizer)
	
	NewBaseOptimizer.optimizerName = optimizerName or "Unknown"
	
	NewBaseOptimizer.calculationFunction = nil
	
	NewBaseOptimizer.resetFunction = nil
	
	return NewBaseOptimizer
	
end

function BaseOptimizer:calculate(learningRate, costFunctionDerivatives)
	
	if (self.calculateFunction) then return self.calculateFunction(learningRate, costFunctionDerivatives) end
	
end

function BaseOptimizer:reset()
	
	if (self.resetFunction) then return self.resetFunction() end
 	
end

function BaseOptimizer:setCalculateFunction(calculateFunction)
	
	self.calculateFunction = calculateFunction
	
end

function BaseOptimizer:setResetFunction(resetFunction)
	
	self.resetFunction = resetFunction
	
end

function BaseOptimizer:getOptimizerName()
	
	return self.optimizerName
	
end

return BaseOptimizer
