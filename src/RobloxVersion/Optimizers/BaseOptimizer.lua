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

local BaseInstance = require(script.Parent.Parent.Cores.BaseInstance)

BaseOptimizer = {}

BaseOptimizer.__index = BaseOptimizer

setmetatable(BaseOptimizer, BaseInstance)

function BaseOptimizer.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}

	local NewBaseOptimizer = BaseInstance.new(parameterDictionary)

	setmetatable(NewBaseOptimizer, BaseOptimizer)

	NewBaseOptimizer:setName("BaseOptimizer")

	NewBaseOptimizer:setClassName("Optimizer")
	
	NewBaseOptimizer.LearningRateValueScheduler = parameterDictionary.LearningRateValueScheduler

	NewBaseOptimizer.CalculateFunction = nil

	NewBaseOptimizer.optimizerInternalParameterArray = {}

	return NewBaseOptimizer

end

function BaseOptimizer:calculate(learningRate, costFunctionDerivativeTensor)

	local CalculateFunction = self.CalculateFunction

	local LearningRateValueScheduler = self.LearningRateValueScheduler

	if (not CalculateFunction) then error("No calculate function for the optimizer!") end

	if LearningRateValueScheduler then learningRate = LearningRateValueScheduler:calculate(learningRate) end

	return self.CalculateFunction(learningRate, costFunctionDerivativeTensor)

end

function BaseOptimizer:setCalculateFunction(CalculateFunction)

	self.CalculateFunction = CalculateFunction

end

function BaseOptimizer:setLearningRateValueScheduler(LearningRateValueScheduler)

	self.LearningRateValueScheduler = LearningRateValueScheduler

end

function BaseOptimizer:getLearningRateValueScheduler()

	return self.LearningRateValueScheduler

end

function BaseOptimizer:getOptimizerInternalParameterArray(doNotDeepCopy)

	if (doNotDeepCopy) then

		return self.optimizerInternalParameterArray

	else

		return self:deepCopyTable(self.optimizerInternalParameterArray)

	end

end

function BaseOptimizer:setOptimizerInternalParameterArray(optimizerInternalParameterArray, doNotDeepCopy)

	if (doNotDeepCopy) then

		self.optimizerInternalParameterArray = optimizerInternalParameterArray

	else

		self.optimizerInternalParameterArray = self:deepCopyTable(optimizerInternalParameterArray)

	end

end

function BaseOptimizer:reset()

	self.optimizerInternalParameterArray = {}

end

return BaseOptimizer
