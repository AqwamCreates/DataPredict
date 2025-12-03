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

local TabularReinforcementLearningBaseModel = {}

TabularReinforcementLearningBaseModel.__index = TabularReinforcementLearningBaseModel

setmetatable(TabularReinforcementLearningBaseModel, BaseInstance)

local defaultDiscountFactor = 0.95

function TabularReinforcementLearningBaseModel.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewTabularReinforcementLearningBaseModel = BaseInstance.new(parameterDictionary)
	
	setmetatable(NewTabularReinforcementLearningBaseModel, TabularReinforcementLearningBaseModel)
	
	NewTabularReinforcementLearningBaseModel:setName("TabularReinforcementLearningBaseModel")

	NewTabularReinforcementLearningBaseModel:setClassName("TabularReinforcementLearningModel")
	
	NewTabularReinforcementLearningBaseModel.Model = parameterDictionary.Model

	NewTabularReinforcementLearningBaseModel.discountFactor = parameterDictionary.discountFactor or defaultDiscountFactor
	
	return NewTabularReinforcementLearningBaseModel
	
end

function TabularReinforcementLearningBaseModel:setDiscountFactor(discountFactor)
	
	self.discountFactor = discountFactor
	
end

function TabularReinforcementLearningBaseModel:getDiscountFactor()
	
	return self.discountFactor
	
end

function TabularReinforcementLearningBaseModel:setStatesList(StatesList)
	
	self.Model:setFeaturesList(StatesList)
	
end

function TabularReinforcementLearningBaseModel:getStatesList()
	
	return self.Model:getFeaturesList()
	
end

function TabularReinforcementLearningBaseModel:setActionsList(ActionsList)
	
	self.Model:setClassesList(ActionsList)
	
end

function TabularReinforcementLearningBaseModel:getActionsList()
	
	return self.Model:getClassesList()
	
end

function TabularReinforcementLearningBaseModel:setCategoricalUpdateFunction(categoricalUpdateFunction)

	self.categoricalUpdateFunction = categoricalUpdateFunction

end

function TabularReinforcementLearningBaseModel:categoricalUpdate(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)

	if (not self.ModelParameters) then self.ModelParameters = self:initializeMatrixBasedOnMode({#self.StatesList, #self.ActionsList}) end
	
	self.categoricalUpdateFunction(previousStateValue, previousAction, rewardValue, currentStateValue, currentAction, terminalStateValue)

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
