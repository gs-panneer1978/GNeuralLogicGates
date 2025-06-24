#pragma once


// Use the C++ bindings for OpenCL
#include "cl.hpp"

#include <vector>
#include <string>
#include "GObject.h"
#include "GTypes.h" // Assuming this contains typedefs like Topology and VectorDouble
#include "GNeuronOpenCL.h" // The OpenCL layer manager class
#include "InterfaceGNeuralNet.h" // The interface for neural networks


/// <summary>
/// Manages a complete neural network running on an OpenCL-capable device (GPU).
/// This class handles the initialization of the OpenCL environment, compilation of kernels,
/// management of network layers, and the orchestration of feedforward and backpropagation passes.
/// </summary>
class GNeuralNetOCL : public InterfaceGNeuralNet, public GObject
{
public:
    /// <summary>
    /// Constructs the network manager and initializes the OpenCL environment.
    /// It will select the first available GPU by default.
    /// </summary>
    /// <param name="topology">A vector defining the network structure (e.g., {input_count, hidden1_count, output_count}).</param>
                                    GNeuralNetOCL();// = default;

                                    GNeuralNetOCL(const Topology& topology, const std::string& file_name = "");

	int                             GetTypeID() const override { return defNeuralNetOCL; }
	/// <summary>
	/// Construct the network from topology data.
	/// </summary>
    void                            buildFromTopology(const Topology& topology);
    /// <summary>
    /// Executes the full feedforward pass on the GPU.
    /// </summary>
    /// <param name="inputVals">A vector of input values for the network.</param>
    void                            feedForward(const VectorDouble& inputVals);

    /// <summary>
    /// Executes the full backpropagation and weight update pass on the GPU.
    /// </summary>
    /// <param name="targetVals">A vector of target values for the output layer.</param>
    void                            backPropagate(const VectorDouble& targetVals);

    /// <summary>
    /// Retrieves the output of the final layer from the GPU back to the host.
    /// This should be called after feedForward().
    /// </summary>
    /// <param name="resultVals">A vector that will be filled with the network's output values.</param>
    void                            getResults(VectorDouble& resultVals) const;

    /// <summary>
    /// Sets the training parameters for the backpropagation step.
    /// </summary>
    void                            SetTrainingParameters(double learningRate, double momentum,
                                        GNeuronOpenCL::OptimizerType optimizer = GNeuronOpenCL::OptimizerType::Momentum,
                                        int activationType = 1, double adam_b1 = 0.9, double adam_b2 = 0.999);
     
	/// <summary>
	/// Sets the activation type for the neurons in the network.
	/// </summary>
    void                            SetActivationType(ENUM_ACTIVATION activationType);

    void                            buildNetwork(const Topology& topology);
    /// <summary>
    /// Saves the network weights to a binary file.
    /// </summary>
    bool                            saveNetwork(const std::string& file_name) const;

    /// <summary>
    /// Loads network weights from a binary file.
    /// </summary>
    bool                            loadNetwork(const std::string& file_name);
	/// <summary>
    /// Resets all layer weights to a new random initialization.
	/// </summary>
    void                            reinitializeWeights();
	/// <summary>
    // ADD THIS: Performs one forward/backward pass on a single batch and returns the loss.
	/// This is useful for training with mini-batches.
	/// </summary>
    double                          trainSingleBatch(const std::vector<double>& input, const std::vector<double>& target);
    /// <summary>
    /// Determines if the device supports fp64 (double precision).
    /// </summary>
    bool                            supports_fp64(const cl::Device& device);

	//for debugging purposes
    const GNeuronOpenCL&            getLayer(size_t index) const { return m_layers[index]; }
	// Returns the topology of the network.
    Topology                        getTopology() { return m_topology; }
	// Sets the learning rate for the network. (Deprecated. demonstration purposes only, use SetTrainingParameters instead)
    void                            SetLearningRate(double learning_rate);
	// Sets the momentum for the network. (Deprecated. demonstration purposes only, use SetTrainingParameters instead)
    void                            SetMomentum(double momentum);
	// Returns the current learning rate.
    double                          GetLearningRate(void) const {  return m_learningRate; }
    double                          GetMomentum(void) const {  return m_momentum;  }
    void                            Display(const std::string& title) const;

    // Helper function to print a vector for debugging
    static void                     printVector(const std::string& title, const std::vector<double>& vec);
    

private:
    // --- Private Helper Methods for Setup ---
    void                            initializeOpenCL();
    void                            createAndBuildProgram(const std::string& kernel_file_path);
    void                            createNetworkLayers(); // OBSOLETE: UPDATED using createNetworkLayersAndBuffers
	void                            createNetworkLayersAndBuffers();


    // --- Core OpenCL Objects ---
    cl::Context                     m_context;
    cl::Device                      m_device;
    cl::CommandQueue                m_queue;
    cl::Program                     m_program;

    // --- Network Structure and Data ---
    Topology                        m_topology;
    std::vector<GNeuronOpenCL>      m_layers;

    // --- GPU Buffers for Network I/O ---
    cl::Buffer                      m_inputBuffer;  // Buffer to hold the initial input values
    cl::Buffer                      m_targetBuffer; // Buffer to hold target values for training

    cl::Buffer                      m_squaredErrorBuffer;  // Buffer for RecentAverageError calculation

	// --- Additional OpenCL Kernels for Network Operations  ---
    cl::Kernel                      m_kernelCalcSquaredError;

    // --- Training Parameters ---
    bool                            m_using_fp64; // flag to store the precision mode
    double                          m_learningRate = 0.00001;
    double                          m_momentum = 0.1;
    double                          m_adam_b1 = 0.9;
    double                          m_adam_b2 = 0.999;

    
    ENUM_ACTIVATION                 m_activationType; // 0=tanh, 1=sigmoid
    GNeuronOpenCL::OptimizerType    m_optimizer = GNeuronOpenCL::OptimizerType::Adam;

	double                          m_recentAverageError = 0.0; // For tracking training performance
};