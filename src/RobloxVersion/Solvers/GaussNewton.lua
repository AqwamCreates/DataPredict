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
	
	setmetatable(NewGaussNewtonSolver, GaussNewtonSolver)
	
	NewGaussNewtonSolver:setName("GaussNewton")
	
	NewGaussNewtonSolver:setCalculateFunction(function(weightMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)
		
		-- Can only cache from linear models since the derivative is a feature matrix. Hence, these values are constant.
		
		local isLinearInput = (not NewGaussNewtonSolver.isNonLinearInput)
		
		local pseudoInverseMatrix = (isLinearInput and NewGaussNewtonSolver.cache)

		if (not pseudoInverseMatrix) then

			local transposedFirstDerivativeMatrix = AqwamTensorLibrary:transpose(firstDerivativeMatrix)

			pseudoInverseMatrix = AqwamTensorLibrary:dotProduct(transposedFirstDerivativeMatrix, firstDerivativeMatrix)

			pseudoInverseMatrix = AqwamTensorLibrary:inverse(pseudoInverseMatrix)
			
			-- If it is non-invertible, then do not return any weight change values as it is likely to be a local minimum.
			
			if (not pseudoInverseMatrix) then return AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(weightMatrix), 0) end

			pseudoInverseMatrix = AqwamTensorLibrary:dotProduct(pseudoInverseMatrix, transposedFirstDerivativeMatrix)
			
			if (isLinearInput) then NewGaussNewtonSolver.cache = pseudoInverseMatrix end

		end

		return AqwamTensorLibrary:dotProduct(pseudoInverseMatrix, firstDerivativeLossMatrix)
		
	end)
	
	return NewGaussNewtonSolver
	
end

return GaussNewtonSolver
