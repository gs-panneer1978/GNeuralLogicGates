// NetworkFactory.h
#pragma once

#include <memory>
#include "InterfaceGNeuralNet.h"
#include "GTypes.h"
#include "DllExport.h" // Our new export macro header
#include "GNeuralNet.h"       // Include the CPU implementation
#include "GNeuralNetOCL.h"    // Include the GPU implementation
#include <iostream>

namespace NetworkFactory
{
    /// <summary>
    /// Checks if a compatible OpenCL GPU device is available on the system.
    /// This is safe to call even if the OpenCL runtime is not installed.
    /// </summary>
    /// <returns>true if an OpenCL GPU is found, false otherwise.</returns>
    GNEURAL_API bool IsOpenCLDeviceAvailable();

    /// <summary>
    /// Creates the most appropriate neural network instance automatically.
	/// used creating a neural network based on the provided topology.
    /// It will create a GNeuralNetOCL if a GPU is available, otherwise it falls back to GNeuralNet (CPU).
    /// </summary>
    /// <param name="topology">The network structure.</param>
    /// <returns>A unique_ptr to the created network object.</returns>
    //GNEURAL_API std::unique_ptr<InterfaceGNeuralNet> CreateNetworkAuto(const Topology& topology, const std::string& file_name = "");

    /// <summary>
    /// Prompts the user via the console to choose which network backend to use (CPU or GPU).
    /// </summary>
    /// <param name="topology">The network structure.</param>
    /// <returns>A unique_ptr to the created network object based on user selection.</returns>
    //GNEURAL_API std::unique_ptr<InterfaceGNeuralNet> CreateNetworkInteractive(const Topology& topology, const std::string& file_name = "");
    
    /// <summary>
   /// Creates a new, randomly initialized neural network from a given topology.
   /// It will automatically select the best backend (GPU or CPU).
   /// </summary>
    GNEURAL_API std::unique_ptr<InterfaceGNeuralNet> CreateNewNetwork(const Topology& topology);

    /// <summary>
    /// Loads a neural network from a saved file. The network structure is determined
    /// by the data within the file itself.
    /// It will automatically create the correct backend (GPU or CPU) that the network was saved with.
    /// </summary>
    GNEURAL_API std::unique_ptr<InterfaceGNeuralNet> LoadNetworkFromFile(const std::string& file_name);
}

