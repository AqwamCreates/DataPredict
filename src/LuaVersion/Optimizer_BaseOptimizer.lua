BaseOptimizer = {}

BaseOptimizer.__index = BaseOptimizer

function BaseOptimizer.new(optimizerName)
	
	local NewBaseOptimizer = {}
	
	setmetatable(NewBaseOptimizer, BaseOptimizer)
	
	NewBaseOptimizer.optimizerName = optimizerName
	
	NewBaseOptimizer.calculationFunction = nil
	
	NewBaseOptimizer.resetFunction = nil
	
	return BaseOptimizer
	
end

function BaseOptimizer:calculate(learningRate, costFunctionDerivatives)
	
	if not self.calculationFunction then return self.calculationFunction(costFunctionDerivatives) end
	
end

function BaseOptimizer:reset()
	
	if not self.resetFunction then return self.resetFunction() end
 	
end

function BaseOptimizer:setCalculationFunction(calculationFunction)
	
	self.calculationFunction = calculationFunction
	
end

function BaseOptimizer:setResetFunction(resetFunction)
	
	self.resetFunction = resetFunction
	
end

function BaseOptimizer:getOptimizerName()
	
	return self.optimizerName
	
end

return BaseOptimizer
