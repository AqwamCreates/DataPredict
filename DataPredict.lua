--[[

	--------------------------------------------------------------------

	Version 1.5.0

	Aqwam's Roblox Deep Learning Library (AR-MDLL)

	Author: Aqwam Harish Aiman
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	DO NOT SELL, RENT, DISTRIBUTE THIS LIBRARY
	
	DO NOT SELL, RENT, DISTRIBUTE MODIFIED VERSION OF THIS LIBRARY
	
	DO NOT CLAIM OWNERSHIP OF THIS LIBRARY
	
	GIVE CREDIT AND SOURCE WHEN USING THIS LIBRARY IF YOUR USAGE FALLS UNDER ONE OF THESE CATEGORIES:
	
		- USED AS A VIDEO OR ARTICLE CONTENT
		- USED AS COMMERCIAL USE OR PUBLIC USE
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/Aqwam-Roblox-Machine-And-Deep-Learning-Library/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------

--]]

local requiredMatrixLibraryVersion = 1.8

local AqwamMatrixLibrary = require(script.AqwamRobloxMatrixLibraryLinker.Value)

local Models = script.Models

local AqwamCustomModels = script.AqwamCustomModels

local Others = script.Others

local Optimizers = script.Optimizers

local ModelsDictionary = {
	
	LinearRegression = require(Models.LinearRegression),
	
	LogisticRegression = require(Models.LogisticRegression),
	
	KMeans = require(Models.KMeans),
	
	SupportVectorMachine = require(Models.SupportVectorMachine),
	
	NaiveBayes = require(Models.NaiveBayes),
	
	ExpectationMaximization = require(Models.ExpectationMaximization),
	
	NeuralNetwork = require(Models.NeuralNetwork),
	
	KMedoids = require(Models.KMedoids),
	
	LogisticRegressionOneVsAll = require(Models.LogisticRegressionOneVsAll),
	
	SupportVectorMachineOneVsAll = require(Models.SupportVectorMachineOneVsAll),
	
	AffinityPropagation = require(Models.AffinityPropagation),
	
	AgglomerativeHierarchical = require(Models.AgglomerativeHierarchical),
	
	DensityBasedSpatialClusteringOfApplicationsWithNoise = require(Models.DensityBasedSpatialClusteringOfApplicationsWithNoise),
	
}

local AqwamCustomModelsDictionary = {
	
	QueuedReinforcementNeuralNetwork = require(AqwamCustomModels.QueuedReinforcementNeuralNetwork),
	
}

local OptimizersDictionary = {
	
	RootMeanSquarePropagation = require(Optimizers.RootMeanSquarePropagation),
	
	Momentum = require(Optimizers.Momentum),
	
	AdaptiveGradient = require(Optimizers.AdaptiveGradient),
	
	AdaptiveMomentEstimation = require(Optimizers.AdaptiveMomentEstimation)
	
}

local OthersDictionary = {
	
	ModelChecking = require(Others.ModelChecking),
	
	GradientDescentModes =  require(Others.GradientDescentModes),
	
	Regularization = require(Others.Regularization),
	
	StringSplitter = require(Others.StringSplitter),
	
	OnlineLearning = require(Others.OnlineLearning)
	
}

local AqwamRobloxMachineLearningLibrary = {}

AqwamRobloxMachineLearningLibrary.Models =  ModelsDictionary

AqwamRobloxMachineLearningLibrary.AqwamCustomModels = AqwamCustomModelsDictionary

AqwamRobloxMachineLearningLibrary.Optimizers = OptimizersDictionary

AqwamRobloxMachineLearningLibrary.Others = OthersDictionary

local function checkVersion()
	
	local matrixLibraryVersion
	
	if (AqwamMatrixLibrary == nil) then error("\n\nMatrixL (or Aqwam's Matrix Library) is not linked to this library. \nPlease read the \"Installation & Usage \" in DataPredict's documentation for installation details. ") end
	
	local success = pcall(function()
		
		matrixLibraryVersion = AqwamMatrixLibrary:getVersion()
		
	end)
	
	if not success then matrixLibraryVersion = -1 end
	
	if (matrixLibraryVersion < requiredMatrixLibraryVersion) then warn("The matrix library is out-of-date. You may encounter some problems.") end
	
end

checkVersion()

return AqwamRobloxMachineLearningLibrary
