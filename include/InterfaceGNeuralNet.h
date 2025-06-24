#pragma once

#include <vector>
#include <string>
#include "GNeuronOpenCL.h"
#include "GTypes.h" // Your type definitions
// Abstract base class defining the common interface for all network types.
class InterfaceGNeuralNet {

public:
    // A virtual destructor is CRITICAL for a polymorphic base class.
    virtual             ~InterfaceGNeuralNet() = default;
	// Get the type ID for runtime type identification.
	//virtual int GetTypeID() const = 0;
	// Set the training parameters for the network.
    virtual void        SetTrainingParameters(double learningRate, double momentum,
                            GNeuronOpenCL::OptimizerType optimizer = GNeuronOpenCL::OptimizerType::Momentum,
                            int activationType = 0, double adam_b1 = 0.9, double adam_b2 = 0.999) = 0;
	// Set the activation type for the network.
	virtual void        SetActivationType(ENUM_ACTIVATION activationType) = 0;
    // Common public methods that both CPU and GPU versions will implement.
    virtual void        feedForward(const VectorDouble& inputVals) = 0;
	// Backpropagate the target values to adjust the weights.
    virtual void        backPropagate(const VectorDouble& targetVals) = 0;
	// Get the results from the network after a feedforward pass.
    virtual void        getResults(VectorDouble& resultVals) const = 0;
	// Get the recent average error for monitoring training progress.
    virtual bool        saveNetwork(const std::string& file_name) const = 0;
	// Load a network from a file, returning true on success.
    virtual bool        loadNetwork(const std::string& file_name) = 0;
    // Retrieve the network topology
    virtual Topology    getTopology() = 0;
	// Get the learning rate value for the network.void		
	virtual void		SetLearningRate(double learning_rate) = 0;
	// Get the momentum value for the network.
	virtual void		SetMomentum(double momentum) = 0;
    // Get the learning rate value for the network.
	virtual double      GetLearningRate(void) const = 0;
	// Get the momentum value for the network.
	virtual double      GetMomentum(void) const = 0;

	// Display the network structure or state.
	virtual void        Display(const std::string& title) const = 0;
    
};

