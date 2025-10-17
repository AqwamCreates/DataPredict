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

local ReinforcementLearningBaseQuickSetup = require(script.Parent.ReinforcementLearningBaseQuickSetup)

DiagonalGaussianPolicyBaseQuickSetup = {}

DiagonalGaussianPolicyBaseQuickSetup.__index = DiagonalGaussianPolicyBaseQuickSetup

setmetatable(DiagonalGaussianPolicyBaseQuickSetup, ReinforcementLearningBaseQuickSetup)

function DiagonalGaussianPolicyBaseQuickSetup.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local NewDiagonalGaussianPolicyBaseQuickSetup = ReinforcementLearningBaseQuickSetup.new(parameterDictionary)
	
	setmetatable(NewDiagonalGaussianPolicyBaseQuickSetup, DiagonalGaussianPolicyBaseQuickSetup)
	
	NewDiagonalGaussianPolicyBaseQuickSetup:setName("DiagonalGaussianPolicyBaseQuickSetup")
	
	NewDiagonalGaussianPolicyBaseQuickSetup:setClassName("DiagonalGaussianPolicyQuickSetup")
	
	return NewDiagonalGaussianPolicyBaseQuickSetup
	
end

return DiagonalGaussianPolicyBaseQuickSetup
