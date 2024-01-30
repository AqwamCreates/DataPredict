--[[

	--------------------------------------------------------------------

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS RESEARCH AND EDUCATION CONTENT
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamMatrixLibraryLinker.Value)

BaseModel = {}

BaseModel.__index = BaseModel

local function deepCopyTable(original, copies)
	
	copies = copies or {}
	
	local originalType = type(original)
	
	local copy
	
	if (originalType == 'table') then
		
		if copies[original] then
			
			copy = copies[original]
			
		else
			
			copy = {}
			
			copies[original] = copy
			
			for originalKey, originalValue in next, original, nil do
				
				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)
				
			end
			
			setmetatable(copy, deepCopyTable(getmetatable(original), copies))
			
		end
		
	else -- number, string, boolean, etc
		
		copy = original
		
	end
	
	return copy
	
end

function BaseModel.new()
	
	local NewBaseModel = {}
	
	setmetatable(NewBaseModel, BaseModel)
	
	NewBaseModel.IsOutputPrinted = true

	NewBaseModel.ModelParameters = nil
	
	NewBaseModel.ModelParametersInitializationMode = "RandomUniformNegativeAndPositive"
	
	NewBaseModel.NumberOfIterationsPerCostCalculation = 1
	
	NewBaseModel.MinimumModelParametersInitializationValue = nil

	NewBaseModel.MaximumModelParametersInitializationValue = nil
	
	NewBaseModel.IterationWaitDuration = nil
	
	NewBaseModel.DataWaitDuration = nil
	
	NewBaseModel.SequenceWaitDuration = nil
	
	NewBaseModel.targetCostUpperBound = 0
	
	NewBaseModel.targetCostLowerBound = 0
	
	NewBaseModel.currentCostToCheckForConvergence = nil
	
	NewBaseModel.currentNumberOfIterationsToCheckIfConverged = 1
	
	NewBaseModel.numberOfIterationsToCheckIfConverged = math.huge
	
	NewBaseModel.AutoResetOptimizers = true

	return NewBaseModel
	
end

function BaseModel:setNumberOfIterationsToCheckIfConverged(numberOfIterations)
	
	self.numberOfIterationsToCheckIfConverged = numberOfIterations or self.numberOfIterationsToCheckIfConverged
	
end

function BaseModel:checkIfConverged(cost)
	
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
		
		self.currentNumberOfIterationsToCheckIfConverged += 1
		
		return false
		
	end
	
	self.currentNumberOfIterationsToCheckIfConverged = 1
	
	self.currentCostToCheckForConvergence = nil
	
	return true
	
end

function BaseModel:setTargetCost(upperBound, lowerBound)

	self.targetCostUpperBound = upperBound or self.targetCostUpperBound
	
	self.targetCostLowerBound = lowerBound or self.targetCostLowerBound

end

function BaseModel:checkIfTargetCostReached(cost)
	
	if (not cost) then return false end
	
	return (cost >= self.targetCostLowerBound) and (cost <= self.targetCostUpperBound)
	
end

function BaseModel:calculateCostWhenRequired(currentNumberOfIteration, costFunction)
	
	if ((currentNumberOfIteration % self.NumberOfIterationsPerCostCalculation) == 0) then 
		
		return costFunction()
		
	else
		
		return nil
		
	end
	
end

function BaseModel:setNumberOfIterationsPerCostCalculation(numberOfIterationsPerCostCalculation)
	
	self.NumberOfIterationsPerCostCalculation = self:getBooleanOrDefaultOption(numberOfIterationsPerCostCalculation, self.NumberOfIterationsPerCostCalculation)
	
end

function BaseModel:setAutoResetOptimizers(option)
	
	self.AutoResetOptimizers = self:getBooleanOrDefaultOption(option, self.AutoResetOptimizers)
	
end

function BaseModel:setWaitDurations(iterationWaitDuration, dataWaitDuration, sequenceWaitDuration)
	
	self.IterationWaitDuration = iterationWaitDuration

	self.DataWaitDuration = dataWaitDuration

	self.SequenceWaitDuration = sequenceWaitDuration
	
end

function BaseModel:baseModelWait(waitDuration)
	
	if (type(waitDuration) == "nil") or (waitDuration == false) then return nil end
	
	if (type(waitDuration) == "number") then
		
		task.wait(waitDuration)
		
	else
		
		task.wait()
		
	end
	
end

function BaseModel:iterationWait()
	
	self:baseModelWait(self.IterationWaitDuration)
	
end

function BaseModel:dataWait()

	self:baseModelWait(self.DataWaitDuration)

end

function BaseModel:sequenceWait()

	self:baseModelWait(self.SequenceWaitDuration)

end

function BaseModel:getModelParameters()
	
	return deepCopyTable(self.ModelParameters)
	
end

function BaseModel:setModelParameters(ModelParameters)
	
	if ModelParameters then
		
		self.ModelParameters = deepCopyTable(ModelParameters) 
		
	end
	
end

function BaseModel:clearModelParameters()
	
	self.ModelParameters = nil
	
end

function BaseModel:printCostAndNumberOfIterations(cost, numberOfIteration)
	
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
	
	if (type(boolean) == "nil") then return defaultBoolean end
		
	return boolean
	
end

function BaseModel:setModelParametersInitializationMode(initializationMode, minimumModelParametersInitializationValue, maximumModelParametersInitializationValue)
	
	self.ModelParametersInitializationMode = initializationMode
	
	self.MinimumModelParametersInitializationValue = minimumModelParametersInitializationValue
	
	self.MaximumModelParametersInitializationValue = maximumModelParametersInitializationValue
	
end

function BaseModel:initializeMatrixBasedOnMode(numberOfRows, numberOfColumns)
	
	local initializationMode = self.ModelParametersInitializationMode
	
	if (initializationMode == "Zero") then
		
		return AqwamMatrixLibrary:createMatrix(numberOfRows, numberOfColumns, 0)
	
	elseif (initializationMode == "Random") then
		
		return AqwamMatrixLibrary:createRandomMatrix(numberOfRows, numberOfColumns, self.MinimumModelParametersInitializationValue, self.MaximumModelParametersInitializationValue)
		
	elseif (initializationMode == "RandomNormalPositive") then
		
		return AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)
		
	elseif (initializationMode == "RandomNormalNegative") then
		
		local RandomNormal = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:multiply(RandomNormal, -1)
		
	elseif (initializationMode == "RandomNormalNegativeAndPositive") then
		
		local RandomNormal1 = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)
		
		local RandomNormal2 = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:subtract(RandomNormal1, RandomNormal2)
		
	elseif (initializationMode == "RandomUniformPositive") then

		return AqwamMatrixLibrary:createRandomUniformMatrix(numberOfRows, numberOfColumns)

	elseif (initializationMode == "RandomUniformNegative") then

		local RandomUniform = AqwamMatrixLibrary:createRandomUniformMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:multiply(RandomUniform, -1)

	elseif (initializationMode == "RandomUniformNegativeAndPositive") then

		local RandomUniform1 = AqwamMatrixLibrary:createRandomUniformMatrix(numberOfRows, numberOfColumns)

		local RandomUniform2 = AqwamMatrixLibrary:createRandomUniformMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:subtract(RandomUniform1, RandomUniform2)
		
	elseif (initializationMode == "HeNormal") then
		
		local variancePart1 = 2 / numberOfRows
		
		local variancePart = math.sqrt(variancePart1)
		
		local RandomNormal = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)
		
		return  AqwamMatrixLibrary:multiply(variancePart, RandomNormal)
		
	elseif (initializationMode == "HeUniform") then

		local variancePart1 = 6 / numberOfRows

		local variancePart = math.sqrt(variancePart1)

		local RandomUniform = AqwamMatrixLibrary:createRandomUniformMatrix(numberOfRows, numberOfColumns)

		return  AqwamMatrixLibrary:multiply(variancePart, RandomUniform) 
		
	elseif (initializationMode == "XavierNormal") then

		local variancePart1 = 2 / (numberOfRows + numberOfColumns)

		local variancePart = math.sqrt(variancePart1)

		local RandomNormal = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:multiply(variancePart, RandomNormal) 

	elseif (initializationMode == "XavierUniform") then

		local variancePart1 = 6 / (numberOfRows + numberOfColumns)

		local variancePart = math.sqrt(variancePart1)

		local RandomUniform = AqwamMatrixLibrary:createRandomUniformMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:multiply(variancePart, RandomUniform)
		
	elseif (initializationMode == "LeCunNormal") then

		local variancePart1 = 1 / numberOfRows

		local variancePart = math.sqrt(variancePart1)

		local RandomNormal = AqwamMatrixLibrary:createRandomNormalMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:multiply(variancePart, RandomNormal) 
		
	elseif (initializationMode == "LeCunUniform") then

		local variancePart1 = 3 / numberOfRows

		local variancePart = math.sqrt(variancePart1)

		local RandomUniform = AqwamMatrixLibrary:createRandomUniformMatrix(numberOfRows, numberOfColumns)

		return AqwamMatrixLibrary:multiply(variancePart, RandomUniform) 

	end
	
end

function BaseModel:destroy()
	
	setmetatable(self, nil)
	
	table.clear(self)
	
	self = nil
	
end

return BaseModel
