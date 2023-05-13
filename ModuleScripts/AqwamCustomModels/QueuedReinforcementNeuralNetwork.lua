local NeuralNetworkModel = require(script.Parent.Parent.Models.NeuralNetwork)

QueuedReinforcementNeuralNetworkModel = {}

QueuedReinforcementNeuralNetworkModel.__index = QueuedReinforcementNeuralNetworkModel

setmetatable(QueuedReinforcementNeuralNetworkModel, NeuralNetworkModel)

local AqwamMatrixLibrary = require(script.Parent.Parent.AqwamRobloxMatrixLibraryLinker.Value)

function QueuedReinforcementNeuralNetworkModel.new(maxNumberOfIterations, learningRate, activationFunction, targetCost)
	
	local NewQueuedReinforcementNeuralNetworkModel = NeuralNetworkModel.new(maxNumberOfIterations, learningRate, activationFunction, targetCost)
	
	setmetatable(NewQueuedReinforcementNeuralNetworkModel, QueuedReinforcementNeuralNetworkModel)
	
	return NewQueuedReinforcementNeuralNetworkModel
	
end

function QueuedReinforcementNeuralNetworkModel:startQueuedReinforcement(rewardValue, punishValue, showPredictedLabel, showIdleWarning, showWaitingForLabelWarning)

	if (self.IsQueuedReinforcementRunning == true) then error("Queued reinforcement is already active!") end

	self:checkIfRewardAndPunishValueAreGiven(rewardValue, punishValue)

	self.FeatureVectorQueue = {}

	self.LabelQueue = {}

	self.PredictedLabelQueue = {}

	self.ForwardPropagationTableQueue = {}

	self.ZTableQueue = {}
	
	self.CostArrayQueue = {}

	self.IsQueuedReinforcementRunning = true

	if (showPredictedLabel == nil) then showPredictedLabel = false else showPredictedLabel = showPredictedLabel end

	if (showIdleWarning == nil) then showIdleWarning = true else showIdleWarning = showIdleWarning end

	if (showWaitingForLabelWarning == nil) then showWaitingForLabelWarning = false else showWaitingForLabelWarning = showWaitingForLabelWarning end

	local waitInterval = 0.1

	local idleDuration = 0

	local waitDuration = 0

	local idleWarningIssued = false

	local labelWarningIssued = false
	
	local isCurrentlyBackpropagating = false
	
	local infinityCostWarningIssued = false
	
	local PreviousModelParameters

	local predictCoroutine = coroutine.create(function()

		repeat

			task.wait(waitInterval)

			idleDuration += waitInterval

			if (idleDuration >= 30) and (idleWarningIssued == false) and (showIdleWarning == true) then 

				warn("The neural network has been idle for more than 30 seconds. Leaving the thread running may use unnecessary resource.") 

				idleWarningIssued = true	

			elseif (#self.FeatureVectorQueue == 0) then continue

			elseif (self.IsQueuedReinforcementRunning == false) then break end

			local forwardPropagateTable, zTable = self:forwardPropagate(self.FeatureVectorQueue[1], self.ModelParameters, self.activationFunction)

			table.insert(self.ForwardPropagationTableQueue, forwardPropagateTable)

			table.insert(self.ZTableQueue, zTable)

			local allOutputsMatrix = forwardPropagateTable[#forwardPropagateTable]

			local predictedLabel = self:getLabelFromOutputVector(allOutputsMatrix, self.ClassesList)

			table.insert(self.PredictedLabelQueue, predictedLabel)

			table.remove(self.FeatureVectorQueue, 1)

			idleDuration = 0

			idleWarningIssued = false

		until (self.IsQueuedReinforcementRunning == false)

	end)


	local reinforcementCoroutine = coroutine.create(function()

		repeat

			task.wait(waitInterval)

			waitDuration += waitInterval

			if (waitDuration >= 30) and (labelWarningIssued == false) and (showWaitingForLabelWarning == true) then

				warn("The neural network has been waiting for a label for more than 30 seconds. Leaving the thread running may use unnecessary resource.") 

				labelWarningIssued = true	

			elseif (#self.LabelQueue == 0) or (#self.PredictedLabelQueue == 0) or (#self.ForwardPropagationTableQueue == 0) or (#self.ZTableQueue == 0) or (isCurrentlyBackpropagating == true) then continue

			elseif (self.IsQueuedReinforcementRunning == false) then break end
			
			isCurrentlyBackpropagating = true

			if (showPredictedLabel == true) then print("Predicted Label: " .. self.PredictedLabelQueue[1] .. "\t\t\tActual Label: " .. self.LabelQueue[1]) end

			local logisticMatrix = self:convertLabelVectorToLogisticMatrix(self.ModelParameters, self.LabelQueue[1], self.ClassesList)

			local forwardPropagationTable = self.ForwardPropagationTableQueue[1]

			local allOutputsMatrix = forwardPropagationTable[#forwardPropagationTable]
			
			local cost = self:calculateCost(allOutputsMatrix, logisticMatrix, 1)
			
			if (cost == math.huge) and (infinityCostWarningIssued == false) then 
				
				warn("The model diverged! Reverting to previous model parameters! Please repeat the experiment again or change the argument values if this warning occurs often.") 
				
				infinityCostWarningIssued = true
				
				self.ModelParameters = PreviousModelParameters
				
			end
			
			table.insert(self.CostArrayQueue, cost)
			
			PreviousModelParameters = self.ModelParameters

			local lossMatrix = AqwamMatrixLibrary:subtract(allOutputsMatrix, logisticMatrix)

			local backwardPropagateTable = self:backPropagate(self.ModelParameters, lossMatrix, self.ZTableQueue[1], self.activationFunction)

			local deltaTable = self:calculateDelta(self.ForwardPropagationTableQueue[1], backwardPropagateTable)
			
			if (self.PredictedLabelQueue[1] == self.LabelQueue[1]) then

				self.ModelParameters = self:gradientDescent(rewardValue, self.ModelParameters, deltaTable, 1)

			else

				self.ModelParameters = self:gradientDescent(-punishValue, self.ModelParameters, deltaTable)

			end
			
			isCurrentlyBackpropagating = false
			
			waitDuration = 0

			labelWarningIssued = false

			infinityCostWarningIssued = false

			table.remove(self.LabelQueue, 1)

			table.remove(self.PredictedLabelQueue, 1)

			table.remove(self.ZTableQueue, 1)

			table.remove(self.ForwardPropagationTableQueue, 1)
			
			task.spawn(function()
				
				for frame = 1, 70 do task.wait() end -- to allow cost to be fetched. Otherwise it will remove it before it can be fetched!
				
				table.remove(self.CostArrayQueue, 1)
				
			end)

		until (self.IsQueuedReinforcementRunning == false)	

	end)

	local resetCoroutine = coroutine.create(function()

		repeat task.wait(waitInterval) until (self.IsQueuedReinforcementRunning == false)

		self.IsQueuedReinforcementRunning = nil

		self.FeatureVectorQueue = nil

		self.LabelQueue = nil

		self.PredictedLabelQueue = nil

		self.ForwardPropagationTableQueue = nil

		self.ZTableQueue = nil

		waitInterval = nil

		idleDuration = nil

		waitDuration = nil

		idleWarningIssued = nil

		labelWarningIssued = nil
		
		isCurrentlyBackpropagating = nil
		
		infinityCostWarningIssued = nil
		
		PreviousModelParameters = nil

	end)

	coroutine.resume(predictCoroutine)

	coroutine.resume(reinforcementCoroutine)

	coroutine.resume(resetCoroutine)

	return predictCoroutine, reinforcementCoroutine, resetCoroutine

end

function QueuedReinforcementNeuralNetworkModel:stopQueuedReinforcement()

	self.IsQueuedReinforcementRunning = false

end

function QueuedReinforcementNeuralNetworkModel:addFeatureVectorToReinforcementQueue(featureVector)

	if (self.IsQueuedReinforcementRunning == nil) or (self.IsQueuedReinforcementRunning == false) then error("Queued reinforcement is not active!") end

	table.insert(self.FeatureVectorQueue, featureVector)

end

function QueuedReinforcementNeuralNetworkModel:addLabelToReinforcementQueue(label)

	if (self.IsQueuedReinforcementRunning == nil) or (self.IsQueuedReinforcementRunning == false) then error("Queued reinforcement is not active!") end

	table.insert(self.LabelQueue, label)

end

function QueuedReinforcementNeuralNetworkModel:returnPredictedLabelFromReinforcementQueue()

	if (self.IsQueuedReinforcementRunning == nil) or (self.IsQueuedReinforcementRunning == false) then error("Queued reinforcement is not active!") end

	return self.PredictedLabelQueue[1]

end

function QueuedReinforcementNeuralNetworkModel:returnCostFromReinforcementQueue()

	if (self.IsQueuedReinforcementRunning == nil) or (self.IsQueuedReinforcementRunning == false) then error("Queued reinforcement is not active!") end

	return self.CostArrayQueue[1]

end

return QueuedReinforcementNeuralNetworkModel
