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

local BaseModel = require(script.Parent.BaseModel)

TabularReinforcementLearningBaseModel = {}

TabularReinforcementLearningBaseModel.__index = TabularReinforcementLearningBaseModel

setmetatable(TabularReinforcementLearningBaseModel, BaseModel)

local defaultLearningRate = 0.1

local defaultDiscountFactor = 0.95

function TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewDeepReinforcementLearningBaseModel = BaseModel.new(parameterDictionary)
	
	setmetatable(NewDeepReinforcementLearningBaseModel, TabularReinforcementLearningBaseModel)
	
	NewDeepReinforcementLearningBaseModel:setName("TabularReinforcementLearningBaseModel")

	NewDeepReinforcementLearningBaseModel:setClassName("TabularReinforcementLearningModel")
	
	NewDeepReinforcementLearningBaseModel.learningRate = parameterDictionary.learningRate or defaultLearningRate

	NewDeepReinforcementLearningBaseModel.discountFactor = parameterDictionary.discountFactor or defaultDiscountFactor
	
	NewDeepReinforcementLearningBaseModel.Optimizer = parameterDictionary.Optimizer
	
	NewDeepReinforcementLearningBaseModel.StatesList = parameterDictionary.StatesList or {}
	
	NewDeepReinforcementLearningBaseModel.ActionsList = parameterDictionary.ActionsList or {}
	
	NewDeepReinforcementLearningBaseModel.ModelParameters = parameterDictionary.ModelParameters
	
	return NewDeepReinforcementLearningBaseModel
	
end

function TabularReinforcementLearningBaseModel:setLearningRate(learningRate)

	self.learningRate = learningRate

end

function TabularReinforcementLearningBaseModel:getLearningRate()

	return self.learningRate

end

function TabularReinforcementLearningBaseModel:setDiscountFactor(discountFactor)
	
	self.discountFactor = discountFactor
	
end

function TabularReinforcementLearningBaseModel:getDiscountFactor()
	
	return self.discountFactor
	
end

function TabularReinforcementLearningBaseModel:setOptimizer(Optimizer)

	self.Optimizer = Optimizer

end

function TabularReinforcementLearningBaseModel:getOptimizer()

	return self.Optimizer

end

function TabularReinforcementLearningBaseModel:predict(stateVector, returnOriginalOutput)
	
	local resultTensor = {}
	
	local StatesList = self.StatesList
	
	local ActionsList = self.ActionsList
	
	local ModelParameters = self.ModelParameters
	
	if (not ModelParameters) then
		
		ModelParameters = self:initializeMatrixBasedOnMode({#StatesList, #ActionsList})
		
		self.ModelParameters = ModelParameters
		
	end
	
	for i, wrappedState in ipairs(stateVector) do
		
		local state = wrappedState[1]
		
		local stateIndex = table.find(StatesList, state)
		
		if (not stateIndex) then error("State \"" .. state ..  "\" does not exist in the states list.") end
		
		resultTensor[i] = ModelParameters[stateIndex]
		
	end
	
	if (returnOriginalOutput) then return resultTensor end
	
	local outputVector = {}
	
	local maximumValueVector = {}
	
	for i, resultVector in ipairs(resultTensor) do
		
		local maximumValue = math.max(table.unpack(resultVector))
		
		local actionIndex = table.find(resultVector, maximumValue)
		
		local action = ActionsList[actionIndex] 
		
		if (not action) then error("Action for action index " .. actionIndex ..  "  does not exist in the actions list.") end
		
		outputVector[i] = {action}
		
		maximumValueVector[i] = {maximumValue}
		
	end

	return outputVector, maximumValueVector

end

function TabularReinforcementLearningBaseModel:setStatesList(StatesList)
	
	self.StatesList = StatesList
	
end

function TabularReinforcementLearningBaseModel:getStatesList()
	
	return self.StatesList
	
end

function TabularReinforcementLearningBaseModel:setActionsList(ActionsList)
	
	self.ActionsList = ActionsList
	
end

function TabularReinforcementLearningBaseModel:getActionsList()
	
	return self.ActionsList
	
end

function TabularReinforcementLearningBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function TabularReinforcementLearningBaseModel:categoricalUpdate(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue)

	if (not self.ModelParameters) then self.ModelParameters = self:initializeMatrixBasedOnMode({#self.StatesList, #self.ActionsList}) end
	
	local isPreviousStateValueTable = true
	
	local isCurrentStateValueTable = true
	
	while isPreviousStateValueTable or isCurrentStateValueTable do
		
		isPreviousStateValueTable = (type(previousStateValue) == "table")
		
		isCurrentStateValueTable = (type(currentStateValue) == "table")
		
		if (isPreviousStateValueTable) then previousStateValue = previousStateValue[1] end
		
		if (isCurrentStateValueTable) then currentStateValue = currentStateValue[1] end
		
	end
	
	self.categoricalUpdateFunction(previousStateValue, action, rewardValue, currentStateValue, terminalStateValue)

end

function TabularReinforcementLearningBaseModel:setEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

function TabularReinforcementLearningBaseModel:episodeUpdate(terminalStateValue)
	
	if (not self.ModelParameters) then self.ModelParameters = self:initializeMatrixBasedOnMode({#self.StatesList, #self.ActionsList}) end

	return self.episodeUpdateFunction(terminalStateValue)

end

function TabularReinforcementLearningBaseModel:setResetFunction(resetFunction)

	self.resetFunction = resetFunction

end

function TabularReinforcementLearningBaseModel:reset()
	
	self.resetFunction()

end

return TabularReinforcementLearningBaseModel
