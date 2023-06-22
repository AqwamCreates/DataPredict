OnlineLearning = {}

OnlineLearning.__index = OnlineLearning

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

local modelDivergedWarningText = "The model diverged! Reverting to previous model parameters! Please repeat the experiment again or change the argument values if this warning occurs often."

function OnlineLearning.new(Model, isOutputRequired, batchSize, isSequentialModel)

	if (Model == nil) then error("Please set a model") end

	if (isOutputRequired == nil) then error("Please set whether or not the model requires a Output") end

	local NewOnlineLearning = {}

	setmetatable(NewOnlineLearning, OnlineLearning)

	NewOnlineLearning.Model = Model

	NewOnlineLearning.InputQueue = {}

	NewOnlineLearning.OutputQueue = {}

	NewOnlineLearning.CostArrayQueue = {}

	NewOnlineLearning.IsOutputRequired = isOutputRequired

	NewOnlineLearning.IsOnlineLearningRunning = false

	NewOnlineLearning.BatchSize = batchSize or 1

	NewOnlineLearning.IsSequentialModel = isSequentialModel or false

	return NewOnlineLearning

end

function OnlineLearning:startNonSequentialTraining()

	local featureMatrix = {}

	local labelVector = {}

	local costArray

	for data = 1, self.BatchSize, 1 do

		table.insert(featureMatrix, self.InputQueue[1][1])

		table.remove(self.InputQueue, 1)

		if (self.IsOutputRequired == true) then

			table.insert(labelVector, {self.OutputQueue[1]}) 

			table.remove(self.OutputQueue, 1)

		end

	end

	costArray = self.Model:train(featureMatrix, labelVector)
	
	if (costArray[1] == math.huge) then

		self.Model:setModelParameters(self.PreviousModelParameters)

		warn(modelDivergedWarningText) 

	end

	return costArray

end

function OnlineLearning:startSequentialTraining()

	local inputSequenceTokenArray

	local outputSequenceTokenArray

	local costArray

	for data = 1, self.BatchSize, 1 do
		
		inputSequenceTokenArray = self.InputQueue[1]

		table.remove(self.InputQueue, 1)

		if (self.IsOutputRequired == true) then 
			
			outputSequenceTokenArray = self.OutputQueue[1]
			
			table.remove(self.OutputQueue, 1)
			
		end
		
		costArray = self.Model:train(inputSequenceTokenArray, outputSequenceTokenArray)
		
		if (costArray[1] == math.huge) then

			self.Model:setModelParameters(self.PreviousModelParameters)

			warn(modelDivergedWarningText) 

		end

	end

	return costArray

end

function OnlineLearning:startOnlineLearning(showFinalCost, showWaitWarning)

	if (self.IsOnlineLearningRunning == true) then error("Online Learning is already active!") end

	self.IsOnlineLearningRunning = true

	if (showFinalCost == nil) then showFinalCost = false else showFinalCost = showFinalCost end

	if (showWaitWarning == nil) then showWaitWarning = true else showWaitWarning = showWaitWarning end

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

			if (waitDuration >= 30) and (waitWarningIssued == false) and (waitWarningIssued == true) then 

				warn("The neural network has been waiting for more than 30 seconds. Leaving the thread running may use unnecessary resource.") 

				waitWarningIssued = true	

			elseif (areBatchesFilled == false) then continue

			elseif (self.IsOnlineLearningRunning == false) then break end

			self.PreviousModelParameters = self.Model:getModelParameters()

			if self.IsSequentialModel then

				costArray = self:startNonSequentialTraining()

			else

				costArray = self:startSequentialTraining()

			end

			cost = costArray[#costArray]

			table.insert(self.CostArrayQueue, costArray)

			if (showFinalCost == true) then print("Final Cost: " .. cost) end

			task.spawn(function()

				for frame = 1, 70 do task.wait() end -- to allow cost to be fetched. Otherwise it will remove it before it can be fetched!

				table.remove(self.CostArrayQueue, 1)

			end)

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

function OnlineLearning:stopOnlineLearning()

	self.IsOnlineLearningRunning = false

end

function OnlineLearning:addInputToOnlineLearningQueue(input)

	if (self.IsOnlineLearningRunning == nil) or (self.IsOnlineLearningRunning == false) then error("Online Learning is not active!") end

	table.insert(self.InputQueue, input)

end

function OnlineLearning:addOutputToOnlineLearningQueue(output)

	if (self.IsOnlineLearningRunning == nil) or (self.IsOnlineLearningRunning == false) then error("Online Learning is not active!") end

	if (typeof(output) ~= "number") and (self.IsSequentialModel == true) then error("Output must be a number!") end

	table.insert(self.OutputQueue, output)

end


function OnlineLearning:returnCostArrayFromOnlineLearningQueue()

	if (self.IsOnlineLearningRunning == nil) or (self.IsOnlineLearningRunning == false) then error("Online Learning is not active!") end

	return self.CostArrayQueue[1]

end

return OnlineLearning
