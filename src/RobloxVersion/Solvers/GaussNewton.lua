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

local AqwamTensorLibrary = require(script.Parent.Parent.AqwamTensorLibraryLinker.Value)

local BaseSolver = require(script.Parent.BaseSolver)

local GaussNewtonSolver = {}

GaussNewtonSolver.__index = GaussNewtonSolver

setmetatable(GaussNewtonSolver, BaseSolver)

function GaussNewtonSolver.new(parameterDictionary)
	
	local NewGaussNewtonSolver = BaseSolver.new(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	setmetatable(NewGaussNewtonSolver, GaussNewtonSolver)
	
	NewGaussNewtonSolver:setName("GaussNewton")
	
	NewGaussNewtonSolver.isLinear = NewGaussNewtonSolver:getValueOrDefaultValue(parameterDictionary.isLinear, false)
	
	NewGaussNewtonSolver:setCalculateFunction(function(matrix, firstDerivativeMatrix, firstDerivativeLossMatrix)
		
		-- Can only cache from linear models since the derivative is a feature matrix. Hence, these values are constant.
		
		local isLinear = NewGaussNewtonSolver.isLinear
		
		local pMatrix = (isLinear and NewGaussNewtonSolver.cache)

		if (not pMatrix) then

			local transposedFirstDerivativeMatrix = AqwamTensorLibrary:transpose(firstDerivativeMatrix)

			pMatrix = AqwamTensorLibrary:dotProduct(transposedFirstDerivativeMatrix, matrix)

			pMatrix = AqwamTensorLibrary:inverse(pMatrix)

			pMatrix = AqwamTensorLibrary:dotProduct(pMatrix, transposedFirstDerivativeMatrix)
			
			if (isLinear) then NewGaussNewtonSolver.cache = pMatrix end

		end

		return AqwamTensorLibrary:dotProduct(pMatrix, firstDerivativeLossMatrix)
		
	end)
	
	return NewGaussNewtonSolver
	
end

return GaussNewtonSolver
