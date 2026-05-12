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

local LevenbergMarquardtSolver = {}

LevenbergMarquardtSolver.__index = LevenbergMarquardtSolver

setmetatable(LevenbergMarquardtSolver, BaseSolver)

local defaultLambda = 1

function LevenbergMarquardtSolver.new(parameterDictionary)
	
	local NewLevenbergMarquardtSolver = BaseSolver.new(parameterDictionary)
	
	setmetatable(NewLevenbergMarquardtSolver, LevenbergMarquardtSolver)
	
	NewLevenbergMarquardtSolver:setName("LevenbergMarquardt")
	
	NewLevenbergMarquardtSolver.lambda = parameterDictionary.lambda or defaultLambda
	
	NewLevenbergMarquardtSolver:setCalculateFunction(function(weightMatrix, inputMatrix, firstDerivativeMatrix, firstDerivativeLossMatrix)
		
		local isLinear = NewLevenbergMarquardtSolver.isLinear
		
		local pseudoInverseMatrix = (isLinear and NewLevenbergMarquardtSolver.cache)

		if (not pseudoInverseMatrix) then

			local jacobianMatrix = inputMatrix

			if (not isLinear) then jacobianMatrix = AqwamTensorLibrary:multiply(jacobianMatrix, firstDerivativeMatrix) end

			local transposedJacobianMatrix = AqwamTensorLibrary:transpose(jacobianMatrix)
			
			local pMatrixDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pseudoInverseMatrix)
			
			local diagonalMatrix = AqwamTensorLibrary:createIdentityTensor(pMatrixDimensionSizeArray, NewLevenbergMarquardtSolver.lambda)
			
			pseudoInverseMatrix = AqwamTensorLibrary:add(pseudoInverseMatrix, diagonalMatrix)

			pseudoInverseMatrix = AqwamTensorLibrary:inverse(pseudoInverseMatrix)
			
			-- If it is non-invertible, then do not return any weight change values as it is likely to be a local minimum.
			
			if (not pseudoInverseMatrix) then return AqwamTensorLibrary:createTensor(AqwamTensorLibrary:getDimensionSizeArray(weightMatrix), 0) end

			pseudoInverseMatrix = AqwamTensorLibrary:dotProduct(pseudoInverseMatrix, transposedJacobianMatrix)
			
			if (isLinear) then NewLevenbergMarquardtSolver.cache = pseudoInverseMatrix end

		end

		return AqwamTensorLibrary:dotProduct(pseudoInverseMatrix, firstDerivativeLossMatrix)
		
	end)
	
	return NewLevenbergMarquardtSolver
	
end

return LevenbergMarquardtSolver
