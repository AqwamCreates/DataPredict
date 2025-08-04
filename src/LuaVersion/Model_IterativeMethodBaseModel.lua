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

local BaseModel = require("Model_BaseModel")

IterativeBaseModel = {}

IterativeBaseModel.__index = IterativeBaseModel

setmetatable(IterativeBaseModel, BaseModel)

function IterativeBaseModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewBaseModel = BaseModel.new(parameterDictionary)

	setmetatable(NewBaseModel, IterativeBaseModel)

	NewBaseModel:setName("IterativeBaseModel")

	NewBaseModel:setClassName("IterativeModel")
	
	NewBaseModel.maximumNumberOfIterations = NewBaseModel:getValueOrDefaultValue(parameterDictionary.maximumNumberOfIterations, 1) 
	
	NewBaseModel.numberOfIterationsPerCostCalculation = NewBaseModel:getValueOrDefaultValue(parameterDictionary.numberOfIterationsPerCostCalculation, 1) 

	NewBaseModel.maximumModelParametersInitializationValue = NewBaseModel:getValueOrDefaultValue(parameterDictionary.maximumModelParametersInitializationValue, nil)
	
	NewBaseModel.minimumModelParametersInitializationValue = NewBaseModel:getValueOrDefaultValue(parameterDictionary.minimumModelParametersInitializationValue, nil)
	
	NewBaseModel.iterationWaitDuration = NewBaseModel:getValueOrDefaultValue(parameterDictionary.iterationWaitDuration, nil)
	
	NewBaseModel.dataWaitDuration = NewBaseModel:getValueOrDefaultValue(parameterDictionary.dataWaitDuration, nil)
	
	NewBaseModel.sequenceWaitDuration = NewBaseModel:getValueOrDefaultValue(parameterDictionary.sequenceWaitDuration, nil)
	
	NewBaseModel.targetCostUpperBound = NewBaseModel:getValueOrDefaultValue(parameterDictionary.targetCostUpperBound, 0)
	
	NewBaseModel.targetCostLowerBound = NewBaseModel:getValueOrDefaultValue(parameterDictionary.targetCostLowerBound, 0)
	
	NewBaseModel.currentCostToCheckForConvergence = NewBaseModel:getValueOrDefaultValue(parameterDictionary.currentCostToCheckForConvergence, nil)
	
	NewBaseModel.currentNumberOfIterationsToCheckIfConverged = NewBaseModel:getValueOrDefaultValue(parameterDictionary.currentNumberOfIterationsToCheckIfConverged, 1)
	
	NewBaseModel.numberOfIterationsToCheckIfConverged = NewBaseModel:getValueOrDefaultValue(parameterDictionary.numberOfIterationsToCheckIfConverged, math.huge)

	return NewBaseModel
	
end

function IterativeBaseModel:setNumberOfIterationsToCheckIfConverged(numberOfIterations)
	
	self.numberOfIterationsToCheckIfConverged = numberOfIterations or self.numberOfIterationsToCheckIfConverged
	
end

function IterativeBaseModel:checkIfConverged(cost)
	
	if (not cost) then return false end
	
	if (not self.currentCostToCheckForConvergence) then
		
		self.currentCostToCheckForConvergence = cost
		
		return false
		
	end
	
	if (self.currentCostToCheckForConvergence ~= cost) then
		
		self.currentNumberOfIterationsToCheckIfConverged = 1
		
		self.currentCostToCheckForConvergence = cost

		return false
		
	end
	
	if (self.currentNumberOfIterationsToCheckIfConverged < self.numberOfIterationsToCheckIfConverged) then
		
		self.currentNumberOfIterationsToCheckIfConverged = self.currentNumberOfIterationsToCheckIfConverged + 1
		
		return false
		
	end
	
	self.currentNumberOfIterationsToCheckIfConverged = 1
	
	self.currentCostToCheckForConvergence = nil
	
	return true
	
end

function IterativeBaseModel:resetConvergenceCheck()
	
	self.currentNumberOfIterationsToCheckIfConverged = 1

	self.currentCostToCheckForConvergence = nil
	
end

function IterativeBaseModel:setTargetCost(upperBound, lowerBound)

	self.targetCostUpperBound = upperBound or self.targetCostUpperBound
	
	self.targetCostLowerBound = lowerBound or self.targetCostLowerBound

end

function IterativeBaseModel:checkIfTargetCostReached(cost)
	
	if (not cost) then return false end
	
	return (cost >= self.targetCostLowerBound) and (cost <= self.targetCostUpperBound)
	
end

function IterativeBaseModel:calculateCostWhenRequired(currentNumberOfIteration, costFunction)
	
	if ((currentNumberOfIteration % self.numberOfIterationsPerCostCalculation) == 0) then 
		
		return costFunction()
		
	else
		
		return nil
		
	end
	
end

function IterativeBaseModel:setNumberOfIterationsPerCostCalculation(numberOfIterationsPerCostCalculation)
	
	self.numberOfIterationsPerCostCalculation = self:getValueOrDefaultValue(numberOfIterationsPerCostCalculation, self.numberOfIterationsPerCostCalculation)
	
end

function IterativeBaseModel:setWaitDurations(iterationWaitDuration, dataWaitDuration, sequenceWaitDuration)
	
	self.iterationWaitDuration = iterationWaitDuration

	self.dataWaitDuration = dataWaitDuration

	self.sequenceWaitDuration = sequenceWaitDuration
	
end

function IterativeBaseModel:baseModelWait(waitDuration)
	
	if (type(waitDuration) == "nil") or (waitDuration == false) then return nil end
	
	if (type(waitDuration) == "number") then
		
		task.wait(waitDuration)
		
	else
		
		task.wait()
		
	end
	
end

function IterativeBaseModel:iterationWait()
	
	self:baseModelWait(self.iterationWaitDuration)
	
end

function IterativeBaseModel:dataWait()

	self:baseModelWait(self.dataWaitDuration)

end

function IterativeBaseModel:sequenceWait()

	self:baseModelWait(self.sequenceWaitDuration)

end

function IterativeBaseModel:printNumberOfIterationsAndCost(numberOfIterations, cost)
	
	if (not self.isOutputPrinted) then return end
	
	print("Iteration: " .. numberOfIterations .. "\t\tCost: " .. cost)
	
end

return IterativeBaseModel
