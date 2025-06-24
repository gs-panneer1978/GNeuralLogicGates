#pragma once
#include <vector>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include "constants.h"
#include "GTypes.h"
#include "GNeuron.h"
#include <cassert>
#include "InterfaceGNeuralNet.h"


class GNeuralNet : public InterfaceGNeuralNet, public GObject
{

public:
		// Default constructor: Initializes an empty network object.
					GNeuralNet(); // = default;
		// Constructor that builds the network from a given topology.
					GNeuralNet(const Topology &topology, const std::string& file_name = "");

		// Get the type ID for runtime type identification.
		int			GetTypeID() const override { return defNet; }
		// Main feedforward method to process input values through the network.
		void		feedForward(const VectorDouble& inputVals) override;
		// Backpropagation method to adjust weights based on target values.
		void		backPropagate(const VectorDouble& targetVals) override;
		// Retrieves the results from the output layer after a feedforward pass.
		void		getResults(VectorDouble& resultVals) const override;
		// Generates softmax output for multi-class classification problems. DEPRECATED: use getResults() instead.
		void		generateSoftMaxOutput(VectorDouble &outputSigma, VectorDouble &results) const ;
		// Reinitializes all layer weights to a new random initialization.
		double		getRecentAverageError(void) const { return m_recentAverageError; };
		// Save network topology & weights to a binary file.
		bool		saveNetwork(const string& file_name) const override;
		// Load network topology & weights from a binary file.
		bool		loadNetwork(const string& file_name) override;
		// Train the network using feedforward, backpropagation and update weights.
		bool		TrainNetwork(void);
		// Set the training parameters for the network.
		void		SetTrainingParameters(double learningRate, double momentum,
										GNeuronOpenCL::OptimizerType optimizer, int activationType, double adam_b1, double adam_b2);
		// Set the activation type for the neurons in the network.
		void		SetActivationType(ENUM_ACTIVATION activationType);
		// Set the learning rate and momentum for the network.
		void		SetLearningRate(double learning_rate);
		// Set the momentum for the network.
		void		SetMomentum(double momentum);
		// Retrieves the learning rate value for the network.
		double      GetLearningRate(void) const {  return m_learningRate;  }
		// Retrieves the momentum value for the network.
		double      GetMomentum(void) const {  return m_momentum;  }
		// Displays the network structure or current state.
		void		Display(const std::string& title) const;
		// Returns the topology of the network.
		Topology	getTopology();
private:
		Topology		m_topology;
		vector<Layer>	m_layers;
		ENUM_ACTIVATION m_activationType;
		
		double			m_learningRate = 0.1; // Default learning rate (Eta)
		double			m_momentum = 0.5; // Default momentum (Alpha)
		double			m_error = 0.0;
		double			m_recentAverageError = 0.0;
		double			m_recentAvergeSmoothingFactor = 0.0;
		// File name for saving/loading the network
		std::string		m_file_name; 
		// Helper function to build the network layers based on the topology
		void			build(const Topology& topology);
};

