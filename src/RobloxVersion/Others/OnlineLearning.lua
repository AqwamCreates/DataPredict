OnlineLearning = {}

OnlineLearning.__index = OnlineLearning

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local modelDivergedWarningText = "The model diverged! Reverting to previous model parameters! Please repeat the experiment again or change the argument values if this warning occurs often."

local onlineLearningNotActiveText = "Online Learning is not active!"

local onlineLearningActiveText = "Online Learning is already active!"

function OnlineLearning.new(Model, isOutputRequired, batchSize)

	if (Model == nil) then error("Please set a model!") end

	if (isOutputRequired == nil) then error("Please set whether or not the model requires a output!") end

	local NewOnlineLearning = {}

	setmetatable(NewOnlineLearning, OnlineLearning)

	NewOnlineLearning.Model = Model

	NewOnlineLearning.InputQueue = {}

	NewOnlineLearning.OutputQueue = {}

	NewOnlineLearning.CostArrayQueue = {}

	NewOnlineLearning.IsOutputRequired = isOutputRequired

	NewOnlineLearning.IsOnlineLearningRunning = false

	NewOnlineLearning.BatchSize = batchSize or 1

	return NewOnlineLearning

end

function OnlineLearning:generateDataset(minimumBatchSize)
	
	local input = {}
	
	local output = {}
	
	for data = 1, minimumBatchSize, 1 do

		table.insert(input, self.InputQueue[1][1])

		table.remove(self.InputQueue, 1)

		if (self.IsOutputRequired == false) then continue end

		table.insert(output, self.OutputQueue[1][1]) 

		table.remove(self.OutputQueue, 1)
		
	end
	
	if (self.IsOutputRequired) then
		
		return input, output
		
	else
		
		return input, nil
		
	end
	
end

function OnlineLearning:autoRemoveCostArrayAfterCertainDuration()
	
	task.spawn(function()

		for frame = 1, 70 do task.wait() end -- to allow cost to be fetched. Otherwise it will remove it before it can be fetched!

		table.remove(self.CostArrayQueue, 1)

	end)
	
end

function OnlineLearning:restorePreviousModelParametersIfCostIsInfinity(cost)
	
	if (cost == math.huge) then

		self.Model:setModelParameters(self.PreviousModelParameters)

		warn(modelDivergedWarningText) 

	end
	
end

function OnlineLearning:start(showFinalCost, showIdleWarning)

	if (self.IsOnlineLearningRunning == true) then error(onlineLearningActiveText) end

	self.IsOnlineLearningRunning = true

	if (showFinalCost == nil) then showFinalCost = false end

	if (showIdleWarning == nil) then showIdleWarning = true end

	local waitInterval = 0.1

	local waitDuration = 0

	local waitWarningIssued = false

	local infinityCostWarningIssued = false

	local areBatchesFilled

	local costArray

	local cost

	local trainCoroutine = coroutine.create(function()

		repeat

			task.wait(waitInterval)

			waitDuration += waitInterval

			areBatchesFilled = (#self.InputQueue >= self.BatchSize) and (not self.IsOutputRequired or (#self.OutputQueue >= self.BatchSize))

			if (waitDuration >= 30) and (waitWarningIssued == false) and (showIdleWarning == true) then 

				warn("The neural network has been idle for more than 30 seconds. Leaving the thread running may use unnecessary resource.") 

				waitWarningIssued = true
				
				continue

			elseif (areBatchesFilled == false) then continue

			elseif (self.IsOnlineLearningRunning == false) then break end

			self.PreviousModelParameters = self.Model:getModelParameters()
			
			local minimumBatchSize = math.min(self.BatchSize, #self.InputQueue)
			
			local input, output = self:generateDataset(minimumBatchSize)
			
			local costArray = self.Model:train(input, output)

			cost = costArray[#costArray]
			
			self:restorePreviousModelParametersIfCostIsInfinity(cost)

			table.insert(self.CostArrayQueue, costArray)

			if (showFinalCost == true) then print("Final Cost: " .. cost) end
			
			self:autoRemoveCostArrayAfterCertainDuration()
			
			waitDuration = 0

		until (self.IsQueuedReinforcementRunning == false)

		self.InputQueue = {}

		self.OutputQueue = {}

		self.CostArrayQueue = {}
		
		self.PreviousModelParameters = nil

		waitInterval = nil

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

	if (self.IsOnlineLearningRunning == nil) or (self.IsOnlineLearningRunning == false) then error(onlineLearningNotActiveText) end

	table.insert(self.InputQueue, input)

end

function OnlineLearning:addOutput(output)

	if (self.IsOnlineLearningRunning == nil) or (self.IsOnlineLearningRunning == false) then error(onlineLearningNotActiveText) end
	
	if (type(output) == "number") then output = {{output}} end

	table.insert(self.OutputQueue, output)

end


function OnlineLearning:returnCostArray()

	if (self.IsOnlineLearningRunning == nil) or (self.IsOnlineLearningRunning == false) then error(onlineLearningNotActiveText) end

	return self.CostArrayQueue[1]

end

return OnlineLearning
