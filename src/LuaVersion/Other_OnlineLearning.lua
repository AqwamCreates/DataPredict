--[[

	--------------------------------------------------------------------

	Aqwam's Machine And Deep Learning Library (DataPredict)

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

local BaseInstance = require("Core_BaseInstance")

OnlineLearning = {}

OnlineLearning.__index = OnlineLearning

setmetatable(OnlineLearning, BaseInstance)

local defaultBatchSize = 1

local defaultWaitInterval = 0.1

local defaultShowFinalCost = true

local defaultIdleDuration = 30

local defaultShowIdleWarning = true

local defaultShowModelDivergenceWarning = true

local modelDivergedWarningText = "The model diverged! Reverting to previous model parameters! Please repeat the experiment again or change the argument values if this warning occurs often."

local onlineLearningNotActiveText = "Online Learning is not active!"

local onlineLearningActiveText = "Online Learning is already active!"

function OnlineLearning.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewOnlineLearning = BaseInstance.new(parameterDictionary)

	setmetatable(NewOnlineLearning, OnlineLearning)
	
	local Model = parameterDictionary.Model
	
	local isOutputRequired = parameterDictionary.isOutputRequired

	if (not Model) then error("Please set a model!") end

	if (not isOutputRequired) then error("Please set whether or not the model requires a output!") end

	NewOnlineLearning.Model = Model
	
	NewOnlineLearning.isOutputRequired = isOutputRequired or {}
	
	NewOnlineLearning.batchSize = parameterDictionary.batchSize or defaultBatchSize
	
	NewOnlineLearning.waitInterval = parameterDictionary.waitInterval or defaultWaitInterval
	
	NewOnlineLearning.showFinalCost = NewOnlineLearning:getValueOrDefaultValue(parameterDictionary.showFinalCost, defaultShowFinalCost)
	
	NewOnlineLearning.idleDuration = parameterDictionary.idleDuration or defaultIdleDuration
	
	NewOnlineLearning.showIdleWarning = NewOnlineLearning:getValueOrDefaultValue(parameterDictionary.showIdleWarning, defaultShowIdleWarning)
	
	NewOnlineLearning.showModelDivergenceWarning = NewOnlineLearning:getValueOrDefaultValue(parameterDictionary.showModelDivergenceWarning, defaultShowModelDivergenceWarning)

	NewOnlineLearning.inputQueue = parameterDictionary.inputQueue or {}

	NewOnlineLearning.outputQueue = parameterDictionary.outputQueue or {}

	NewOnlineLearning.costArrayQueue = parameterDictionary.costArrayQueue or {}

	NewOnlineLearning.isOnlineLearningRunning = false

	return NewOnlineLearning

end

function OnlineLearning:generateDataset(minimumBatchSize)
	
	local isOutputRequired = self.isOutputRequired
	
	local inputQueue = self.inputQueue

	local outputQueue = self.outputQueue
	
	local input = {}
	
	local output = {}
	
	for data = 1, minimumBatchSize, 1 do

		table.insert(input,inputQueue [1][1])

		table.remove(inputQueue, 1)

		if (isOutputRequired) then
			
			table.insert(output, outputQueue[1][1]) 

			table.remove(outputQueue, 1)
			
		end
		
	end
	
	if (isOutputRequired) then
		
		return input, output
		
	else
		
		return input, nil
		
	end
	
end

function OnlineLearning:autoRemoveCostArrayAfterCertainDuration()
	
	task.spawn(function()

		for frame = 1, 70 do task.wait() end -- to allow cost to be fetched. Otherwise it will remove it before it can be fetched!

		table.remove(self.costArrayQueue, 1)

	end)
	
end

function OnlineLearning:restorePreviousModelParametersIfCostIsInfinity(cost)
	
	if (cost == math.huge) then

		self.Model:setModelParameters(self.PreviousModelParameters)

		if (self.showModelDivergenceWarning) then warn(modelDivergedWarningText) end

	end
	
end

function OnlineLearning:start()

	if (self.isOnlineLearningRunning) then error(onlineLearningActiveText) end

	self.isOnlineLearningRunning = true
	
	local Model = self.Model
	
	local isOutputRequired = self.isOutputRequired
	
	local batchSize = self.batchSize
	
	local showFinalCost = self.showFinalCost
	
	local idleDuration = self.idleDuration
	
	local showIdleWarning = self.showIdleWarning
	
	local inputQueue = self.inputQueue
	
	local outputQueue = self.outputQueue
	
	local costArrayQueue = self.costArrayQueue

	local waitInterval = self.waitInterval

	local waitDuration = 0

	local waitWarningIssued = false

	local infinityCostWarningIssued = false

	local areBatchesFilled

	local costArray

	local cost

	local trainCoroutine = coroutine.create(function()

		repeat

			task.wait(waitInterval)

			waitDuration = waitDuration + waitInterval

			areBatchesFilled = (#inputQueue >= batchSize) and (not isOutputRequired or (#outputQueue >= batchSize))

			if (waitDuration >= idleDuration) and (not waitWarningIssued) and (showIdleWarning) then 

				warn("The online learning model has been idle for more than " .. idleDuration .. " seconds. Leaving the thread running may use unnecessary resource.") 

				waitWarningIssued = true
				
				continue

			elseif (not areBatchesFilled) then continue

			elseif (not self.isOnlineLearningRunning) then break end

			self.PreviousModelParameters = Model:getModelParameters()
			
			local minimumBatchSize = math.min(batchSize, #inputQueue)
			
			local input, output = self:generateDataset(minimumBatchSize)
			
			local costArray = Model:train(input, output)

			cost = costArray[#costArray]
			
			self:restorePreviousModelParametersIfCostIsInfinity(cost)

			table.insert(costArrayQueue, costArray)

			if (showFinalCost) then print("Final Cost: " .. cost) end
			
			self:autoRemoveCostArrayAfterCertainDuration()
			
			waitDuration = 0

		until (not self.isOnlineLearningRunning)

		self.inputQueue = {}

		self.outputQueue = {}

		self.costArrayQueue = {}
		
		self.PreviousModelParameters = nil

		waitDuration = nil

		infinityCostWarningIssued = nil

	end)

	coroutine.resume(trainCoroutine)

	return trainCoroutine

end

function OnlineLearning:stop()

	self.IsOnlineLearningRunning = false

end

function OnlineLearning:addInput(input)

	if (not self.isOnlineLearningRunning) then error(onlineLearningNotActiveText) end
	
	if (type(input) == "number") then input = {{input}} end

	table.insert(self.inputQueue, input)

end

function OnlineLearning:addOutput(output)

	if (not self.isOnlineLearningRunning) then error(onlineLearningNotActiveText) end
	
	if (type(output) == "number") then output = {{output}} end

	table.insert(self.outputQueue, output)

end

function OnlineLearning:returnCostArray()

	if (not self.isOnlineLearningRunning) then error(onlineLearningNotActiveText) end

	return self.costArrayQueue[1]

end

return OnlineLearning
