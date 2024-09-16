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

local AqwamMatrixLibrary = require("AqwamMatrixLibrary")

DiagonalGaussianPolicyQuickSetup = {}

DiagonalGaussianPolicyQuickSetup.__index = DiagonalGaussianPolicyQuickSetup

local defaultNumberOfReinforcementsPerEpisode = 500

function DiagonalGaussianPolicyQuickSetup.new(numberOfReinforcementsPerEpisode)
	
	local NewDiagonalGaussianPolicyQuickSetup = {}
	
	setmetatable(NewDiagonalGaussianPolicyQuickSetup, DiagonalGaussianPolicyQuickSetup)
	
	NewDiagonalGaussianPolicyQuickSetup.isOutputPrinted = true
	
	NewDiagonalGaussianPolicyQuickSetup.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or defaultNumberOfReinforcementsPerEpisode
	
	NewDiagonalGaussianPolicyQuickSetup.Model = nil
	
	NewDiagonalGaussianPolicyQuickSetup.previousFeatureVector = nil
	
	NewDiagonalGaussianPolicyQuickSetup.currentNumberOfReinforcements = 0

	NewDiagonalGaussianPolicyQuickSetup.currentNumberOfEpisodes = 0
	
	NewDiagonalGaussianPolicyQuickSetup.updateFunction = nil
	
	NewDiagonalGaussianPolicyQuickSetup.episodeUpdateFunction = nil
	
	return NewDiagonalGaussianPolicyQuickSetup
	
end

function DiagonalGaussianPolicyQuickSetup:setParameters(numberOfReinforcementsPerEpisode)
	
	self.numberOfReinforcementsPerEpisode = numberOfReinforcementsPerEpisode or self.numberOfReinforcementsPerEpisode
	
end

function DiagonalGaussianPolicyQuickSetup:extendUpdateFunction(updateFunction)

	self.updateFunction = updateFunction

end

function DiagonalGaussianPolicyQuickSetup:extendEpisodeUpdateFunction(episodeUpdateFunction)

	self.episodeUpdateFunction = episodeUpdateFunction

end

local function getBooleanOrDefaultOption(boolean, defaultBoolean)

	if (type(boolean) == "nil") then return defaultBoolean end

	return boolean

end

function DiagonalGaussianPolicyQuickSetup:setPrintOutput(option)

	self.isOutputPrinted = getBooleanOrDefaultOption(option, self.isOutputPrinted)

end

function DiagonalGaussianPolicyQuickSetup:reinforce(currentFeatureVector, actionStandardDeviationVector, rewardValue)

	if (self.Model == nil) then error("No model!") end
	
	local currentNumberOfReinforcements = self.currentNumberOfReinforcements
	
	local currentNumberOfEpisodes = self.currentNumberOfEpisodes
	
	local previousFeatureVector = self.previousFeatureVector
	
	local Model = self.Model
	
	local updateFunction = self.updateFunction
	
	local randomProbability = Random.new():NextNumber()
	
	local actionMeanVector = Model:predict(currentFeatureVector, true)

	if (previousFeatureVector) then
		
		currentNumberOfReinforcements = currentNumberOfReinforcements + 1

		Model:diagonalGaussianUpdate(previousFeatureVector, actionMeanVector, actionStandardDeviationVector, rewardValue, currentFeatureVector)
		
		if (updateFunction) then updateFunction() end

	end

	if (currentNumberOfReinforcements >= self.numberOfReinforcementsPerEpisode) then
		
		local episodeUpdateFunction = self.episodeUpdateFunction
		
		currentNumberOfReinforcements = 0
		
		currentNumberOfEpisodes = currentNumberOfEpisodes + 1

		Model:episodeUpdate()
		
		if episodeUpdateFunction then episodeUpdateFunction() end

	end
	
	self.currentNumberOfReinforcements = currentNumberOfReinforcements
	
	self.currentNumberOfEpisodes = currentNumberOfEpisodes
	
	self.previousFeatureVector = currentFeatureVector

	if (self.isOutputPrinted) then print("Episode: " .. currentNumberOfEpisodes .. "\t\tReinforcement Count: " .. currentNumberOfReinforcements) end

	return actionMeanVector

end

function DiagonalGaussianPolicyQuickSetup:setModel(Model)

	self.Model = Model

end

function DiagonalGaussianPolicyQuickSetup:getCurrentNumberOfEpisodes()

	return self.currentNumberOfEpisodes

end

function DiagonalGaussianPolicyQuickSetup:getCurrentNumberOfReinforcements()

	return self.currentNumberOfReinforcements

end

function DiagonalGaussianPolicyQuickSetup:getModel()
	
	return self.Model
	
end

function DiagonalGaussianPolicyQuickSetup:reset()
	
	self.currentNumberOfReinforcements = 0

	self.currentNumberOfEpisodes = 0

	self.previousFeatureVector = nil
	
	local Model = self.Model
	
	if (Model) then Model:reset() end
	
end

return DiagonalGaussianPolicyQuickSetup