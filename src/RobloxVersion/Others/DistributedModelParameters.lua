local ModelParametersMerger = require(script.Parent.Parent.Others.ModelParametersMerger)

DistributedModelParameters = {}

DistributedModelParameters.__index = DistributedModelParameters

local defaultTotalNumberOfChildModelUpdatesToUpdateMainModel = 100

function DistributedModelParameters.new(totalNumberOfChildModelUpdatesToUpdateMainModel)
	
	local NewDistributedModelParameters = {}
	
	setmetatable(NewDistributedModelParameters, DistributedModelParameters)
	
	NewDistributedModelParameters.totalNumberOfChildModelUpdatesToUpdateMainModel = totalNumberOfChildModelUpdatesToUpdateMainModel or defaultTotalNumberOfChildModelUpdatesToUpdateMainModel
	
	NewDistributedModelParameters.currentTotalNumberOfChildModelUpdatesToUpdateMainModel = 0
	
	NewDistributedModelParameters.ModelArray = {}
	
	NewDistributedModelParameters.isDistributedLearningRunning = false
	
	NewDistributedModelParameters.ModelParametersMerger = nil
	
	return NewDistributedModelParameters
	
end

function DistributedModelParameters:setParameters(totalNumberOfChildModelUpdatesToUpdateMainModel)
	
	self.totalNumberOfChildModelUpdatesToUpdateMainModel = totalNumberOfChildModelUpdatesToUpdateMainModel or self.totalNumberOfChildModelUpdatesToUpdateMainModel
	
end

function DistributedModelParameters:addModel(Model)
	
	if not Model then error("Model is empty!") end

	table.insert(self.ModelArray, Model)
	
end

function DistributedModelParameters:setModelParametersMerger(ModelParametersMerger)
	
	self.ModelParametersMerger = ModelParametersMerger or self.ModelParametersMerger
	
end

function DistributedModelParameters:train(featureVector, labelVector, modelNumber)

	self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel += 1

	local Model = self.ModelArray[modelNumber]

	if (not Model) then error("No model!") end

	return Model:train(featureVector, labelVector)

end

function DistributedModelParameters:predict(featureVector, returnOriginalOutput, modelNumber)

	local Model = self.ModelArray[modelNumber]

	if not Model then error("No model!") end

	return Model:predict(featureVector, returnOriginalOutput)

end

function DistributedModelParameters:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput, modelNumber)
	
	self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel += 1
	
	local Model = self.ModelArray[modelNumber]
	
	if (not Model) then error("No model!") end
	
	return Model:reinforce(currentFeatureVector, rewardValue, returnOriginalOutput)
	
end

function DistributedModelParameters:setMainModelParameters(MainModelParameters)
	
	self.MainModelParameters = MainModelParameters
	
end

function DistributedModelParameters:getMainModelParameters()
	
	return self.MainModelParameters
	
end

function DistributedModelParameters:getCurrentTotalNumberOfChildModelUpdatesToUpdateMainModel()
	
	return self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel
	
end

function DistributedModelParameters:start()
	
	if (self.isDistributedLearningRunning == true) then error("The model is already running!") end
	
	self.isDistributedLearningRunning = true
	
	local trainCoroutine = coroutine.create(function()

		repeat
			
			task.wait()
			
			if (self.ModelParametersMerger == nil) then warn("No model parameters merger!") continue end
			
			if (self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel < self.totalNumberOfChildModelUpdatesToUpdateMainModel) then continue end
			
			self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel = 0
			
			local ModelParametersArray = {}
			
			for _, Model in ipairs(self.ModelArray) do table.insert(ModelParametersArray, Model:getModelParameters()) end
			
			local MainModelParameters = self.ModelParametersMerger:merge(table.unpack(ModelParametersArray))
			
			for _, Model in ipairs(self.ModelArray) do Model:setModelParameters(MainModelParameters) end
			
			self.MainModelParameters = MainModelParameters

		until (self.isDistributedLearningRunning == false)

	end)

	coroutine.resume(trainCoroutine)

	return trainCoroutine
		
end

function DistributedModelParameters:stop()
	
	self.isDistributedLearningRunning = false
	
end

function DistributedModelParameters:reset()
	
	self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel = 0
	
end

function DistributedModelParameters:destroy()

	setmetatable(self, nil)

	table.clear(self)

	self = nil

end

return DistributedModelParameters
