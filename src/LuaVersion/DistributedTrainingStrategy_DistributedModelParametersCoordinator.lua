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

local BaseInstance = require("Core_BaseInstance")

DistributedModelParametersCoordinator = {}

DistributedModelParametersCoordinator.__index = DistributedModelParametersCoordinator

setmetatable(DistributedModelParametersCoordinator, BaseInstance)

local defaultTotalNumberOfChildModelUpdatesToUpdateMainModel = 10

local defaultCurrentTotalNumberOfChildModelUpdatesToUpdateMainModel = 0

function DistributedModelParametersCoordinator.new(parameterDictionary)
	
	local NewDistributedModelParametersCoordinator = BaseInstance.new(parameterDictionary)
	
	setmetatable(NewDistributedModelParametersCoordinator, DistributedModelParametersCoordinator)
	
	NewDistributedModelParametersCoordinator:setName("DistributedModelParametersCoordinator")
	
	NewDistributedModelParametersCoordinator:setClassName("DistributedModelParametersCoordinator")
	
	NewDistributedModelParametersCoordinator.totalNumberOfChildModelUpdatesToUpdateMainModel = parameterDictionary.totalNumberOfChildModelUpdatesToUpdateMainModel or defaultTotalNumberOfChildModelUpdatesToUpdateMainModel
	
	NewDistributedModelParametersCoordinator.currentTotalNumberOfChildModelUpdatesToUpdateMainModel = parameterDictionary.currentTotalNumberOfChildModelUpdatesToUpdateMainModel or defaultCurrentTotalNumberOfChildModelUpdatesToUpdateMainModel
	
	NewDistributedModelParametersCoordinator.ModelParametersMerger = parameterDictionary.ModelParametersMerger
	
	NewDistributedModelParametersCoordinator.ModelParametersArray = parameterDictionary.ModelParametersArray or {}
	
	NewDistributedModelParametersCoordinator.isDistributedLearningRunning = false
	
	return NewDistributedModelParametersCoordinator
	
end

function DistributedModelParametersCoordinator:addModelParameters(ModelParameters)
	
	if (not ModelParameters) then error("No model parameters!") end
	
	self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel = self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel + 1

	table.insert(self.ModelParametersArray, ModelParameters)
	
end

function DistributedModelParametersCoordinator:setModelParametersMerger(ModelParametersMerger)
	
	self.ModelParametersMerger = ModelParametersMerger or self.ModelParametersMerger
	
end

function DistributedModelParametersCoordinator:setMainModelParameters(MainModelParameters)
	
	self.MainModelParameters = MainModelParameters
	
end

function DistributedModelParametersCoordinator:getMainModelParameters()
	
	return self.MainModelParameters
	
end

function DistributedModelParametersCoordinator:getCurrentTotalNumberOfChildModelUpdatesToUpdateMainModel()
	
	return self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel
	
end

function DistributedModelParametersCoordinator:start()
	
	if (self.isDistributedLearningRunning) then error("It is already running.") end
	
	self.isDistributedLearningRunning = true
	
	local modelParameterChangeCoroutine = coroutine.create(function()
		
		local ModelParametersArray = self.ModelParametersArray

		repeat
			
			task.wait()
			
			local totalNumberOfChildModelUpdatesToUpdateMainModel = self.totalNumberOfChildModelUpdatesToUpdateMainModel
			
			if (self.ModelParametersMerger) then 
				
				if (self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel >= totalNumberOfChildModelUpdatesToUpdateMainModel) then
					
					self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel = 0

					local CurrentModelParametersArray = {}

					for i = 1, totalNumberOfChildModelUpdatesToUpdateMainModel, 1 do

						table.insert(CurrentModelParametersArray, ModelParametersArray[i])

						table.remove(ModelParametersArray, i)

					end
					
					self.MainModelParameters = self.ModelParametersMerger:merge(CurrentModelParametersArray)

				end
				
			else
				
				warn("No model parameters merger.") 
				
			end

		until (not self.isDistributedLearningRunning)

	end)

	coroutine.resume(modelParameterChangeCoroutine)

	return modelParameterChangeCoroutine
		
end

function DistributedModelParametersCoordinator:stop()
	
	self.isDistributedLearningRunning = false
	
end

function DistributedModelParametersCoordinator:reset()
	
	self.currentTotalNumberOfChildModelUpdatesToUpdateMainModel = 0
	
end

return DistributedModelParametersCoordinator
