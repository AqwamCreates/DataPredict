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

local AqwamTensorLibrary = require("AqwamTensorLibrary")

local BaseSolver = require("Solver_BaseSolver")

local GradientSolver = {}

GradientSolver.__index = GradientSolver

setmetatable(GradientSolver, BaseSolver)

function GradientSolver.new(parameterDictionary)
	
	local NewGradientSolver = BaseSolver.new(parameterDictionary)
	
	setmetatable(NewGradientSolver, GradientSolver)
	
	NewGradientSolver:setName("Gradient")
	
	NewGradientSolver:setCalculateFunction(function(weightMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)
		
		-- Can only cache from linear models since the derivative is a feature matrix. Hence, these values are constant.
		
		local isLinearInput = (not NewGradientSolver.isNonLinearInput)

		local transposedFirstDerivativeMatrix = (isLinearInput and NewGradientSolver.cache)
		
		if (not transposedFirstDerivativeMatrix) then
			
			transposedFirstDerivativeMatrix = AqwamTensorLibrary:transpose(firstDerivativeMatrix)
			
			if (isLinearInput) then NewGradientSolver.cache = transposedFirstDerivativeMatrix end
			
		end
		
		return AqwamTensorLibrary:dotProduct(transposedFirstDerivativeMatrix, firstDerivativeLossMatrix)
		
	end)
	
	return NewGradientSolver
	
end

return GradientSolver
