

// Forward declaration if your main network class manages these
#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <stdexcept> // For std::runtime_error
#include <random>
#include <cmath>
// Ensure the OpenCL C++ bindings are included correctly
//#include <CL/opencl.h> // Correct header for OpenCL C++ bindings

#include "constants.h"
#include "cl.hpp"
#include "GTypes.h" 
#include "GObject.h" 


// Forward declaration if your main network class manages these
//class GNeuralNetOCL;

/// <summary>
/// Manages a single layer of neurons on an OpenCL device.
/// This class is responsible for holding the GPU memory buffers (weights, outputs, gradients)
/// for one layer and for executing the OpenCL kernels that operate on this data.
/// </summary>
//GNEURAL_API class GNeuronOpenCL; // Forward declaration for the layer class

class GNeuronOpenCL : public GObject
{
public:
	
    // Default constructor for factory use  
    /// <summary>
    /// Initializes the layer, creates OpenCL kernels, and allocates memory buffers on the device.
    /// </summary>
    GNeuronOpenCL()  
        : m_context(*(cl::Context*)nullptr),  
            m_device(*(cl::Device*)nullptr),  
            m_queue(*(cl::CommandQueue*)nullptr),  
            m_program(*(cl::Program*)nullptr),  
            m_numNeurons(0),  
            m_numInputs(0),
			m_using_fp64(true),  // Default to true, will be set in the real constructor
			m_precision_size(64), // Default to double precision size
		    m_activationType(SIGMOID)  // Default activation type (0=tanh)  
    {}  
    /// <summary>
    /// Initializes the layer, creates OpenCL kernels, and allocates memory buffers on the device.
    /// </summary>
    /// <param name="context">The shared OpenCL context.</param>
    /// <param name="device">The target OpenCL device (GPU).</param>
    /// <param name="queue">The command queue for the device.</param>
    /// <param name="program">The compiled OpenCL program containing the neural net kernels.</param>
    /// <param name="numNeurons">The number of neurons in this layer.</param>
    /// <param name="numInputs">The number of inputs to each neuron (i.e., the size of the previous layer).</param>
    GNeuronOpenCL(cl::Context& context, cl::Device& device, cl::CommandQueue& queue, cl::Program& program,
        size_t numNeurons, size_t numInputs, bool use_fp64);
    /// <summary>
    /// Destructor automatically releases OpenCL resources.
    /// </summary>
    ~GNeuronOpenCL();

	/// <summary>
    /// Returns the type ID of this object, used for runtime type identification.
    /// </summary>
    /// <returns>The type ID for GNeuronOpenCL.</returns>
	/// <remarks>This is used to identify the object type in a polymorphic context.</remarks>
	int GetTypeID() const override { return defNeuronBaseOCL; }
    /// <summary>
    /// Enum to specify which weight update algorithm to use.
    /// </summary>
    enum class OptimizerType {
        Momentum,
        Adam
    };
    /// <summary>
	/// Initializes the OpenCL buffers and kernels for this layer.   
    /// </summary>
    void                        initializeBuffers();
    /// <summary>
    /// Executes the FeedForward kernel for this layer.
    /// </summary>
    /// <param name="prevLayerOutputBuffer">A reference to the output buffer of the preceding layer.</param>
    /// <param name="activationType">The activation function to use (0=tanh, 1=sigmoid).</param>
    void                        feedForward(const cl::Buffer& prevLayerOutputBuffer);

    /// <summary>
    /// Executes the CaclOutputGradient kernel. This should only be called on the final output layer.
    /// </summary>
    /// <param name="targetValuesBuffer">A buffer containing the target values for training.</param>
    /// <param name="activationType">The activation function that was used during feedForward.</param>
    void                        calcOutputGradients(const cl::Buffer& targetValuesBuffer);

    /// <summary>
    /// Executes the CaclHiddenGradient kernel.
    /// </summary>
    /// <param name="nextLayer">A reference to the next layer in the network (the one closer to the output).</param>
    /// <param name="activationType">The activation function that was used during feedForward.</param>
    void                        calcHiddenGradients(const GNeuronOpenCL& nextLayer);

    /// <summary>
    /// Executes the appropriate weight update kernel (Momentum or Adam).
    /// </summary>
    /// <param name="prevLayerOutputBuffer">The output buffer of the preceding layer (which are the inputs to this layer).</param>
    /// <param name="optimizer">The optimizer algorithm to use.</param>
    /// <param name="learningRate">The learning rate (eta).</param>
    /// <param name="momentum">The momentum factor (alpha), used only for the Momentum optimizer.</param>
    /// <param name="b1">Beta1 parameter, used only for the Adam optimizer.</param>
    /// <param name="b2">Beta2 parameter, used only for the Adam optimizer.</param>
    void                            updateInputWeights(const cl::Buffer& prevLayerOutputBuffer, OptimizerType optimizer,
                                        double learningRate, double momentum, double b1, double b2);

    // --- Helper and Accessor Methods ---

    /// <summary>
    /// Copies weight data from the host (CPU) to the device (GPU) buffer.
    /// </summary>
    /// <param name="weights">A vector containing the weights to be written.</param>
    void                            writeWeightsToDevice(const std::vector<double>& weights);

    /// <summary>
    /// Reads the layer's output values from the device (GPU) to the host (CPU).
    /// </summary>
    /// <returns>A vector containing the output values of all neurons in the layer.</returns>
    std::vector<double>             readOutputsFromDevice() const;

    /// <summary>
    /// Provides access to the layer's output buffer, needed by the next layer for its feedForward pass.
    /// </summary>
    const cl::Buffer&               getOutputBuffer() const { return m_bufferOutputs; }

    /// <summary>
    /// Provides access to the layer's weight buffer, needed by the previous layer for backpropagation.
    /// </summary>
    const cl::Buffer&               getWeightBuffer() const { return m_bufferWeights; }

    /// <summary>
    /// Provides access to the layer's gradient buffer, needed by the previous layer for backpropagation.
    /// </summary>
    const cl::Buffer&               getGradientBuffer() const { return m_bufferGradients; }
    /// <summary>
	/// Gets the number of inputs to this layer (i.e., the number of neurons in the previous layer).
    /// </summary>
	size_t						    getInputCount() const { return m_numInputs; }     
    /// <summary>
    /// Gets the number of neurons in this layer.
    /// </summary>
    size_t                          getNeuronCount() const { return m_numNeurons; }

	void                            SetActivationType(ENUM_ACTIVATION activationType) { m_activationType = activationType; }
    void                            Display(const std::string& title) const;

    
    //--- DEBUGGING HELPERS to see GPU's Brain ---
    std::vector<double>             readGradientsFromDevice() const;
    std::vector<double>             readWeightsFromDevice() const;
    
private:
    // --- Private Helper Methods ---
    
    void                            createKernels();

    // --- Core OpenCL Objects (held by reference, managed externally) ---
    cl::Context&                    m_context;
    cl::Device&                     m_device;
    cl::CommandQueue&               m_queue;
    cl::Program&                    m_program;

    // --- Layer Dimensions ---
    size_t                          m_numNeurons;
    size_t                          m_numInputs; // Number of neurons in the previous layer

    // --- OpenCL Kernels for this Layer ---
    cl::Kernel                      m_kernelFeedForward;
    cl::Kernel                      m_kernelCalcOutputGradient;
    cl::Kernel                      m_kernelCalcHiddenGradient;
    cl::Kernel                      m_kernelUpdateWeightsMomentum;
    cl::Kernel                      m_kernelUpdateWeightsAdam;
    

    // --- OpenCL Memory Buffers on the GPU for this Layer's Data ---
    cl::Buffer                      m_bufferWeights;         // Corresponds to matrix_w
    cl::Buffer                      m_bufferOutputs;         // Corresponds to matrix_o
    cl::Buffer                      m_bufferGradients;       // Corresponds to matrix_g/matrix_ig
    cl::Buffer                      m_bufferDeltaWeights;    // For Momentum (matrix_dw)
    cl::Buffer                      m_bufferAdamM;           // For Adam optimizer (matrix_m)
    cl::Buffer                      m_bufferAdamV;           // For Adam optimizer (matrix_v)
	

    bool                            m_using_fp64;
    size_t                          m_precision_size; // Will be sizeof(double) or sizeof(float)
    ENUM_ACTIVATION                 m_activationType; // Default activation type
};

