#pragma once

// DLL export macro
#ifdef GNEURAL_EXPORTS
#define GNEURAL_API __declspec(dllexport)
#else
#define GNEURAL_API __declspec(dllimport)
#endif

#include "GTypes.h"
#include "InterfaceGNeuralNet.h" // Use the common interface, not the concrete class
#include <vector>
#include <memory> // Required for std::unique_ptr

// This class is the main exported interface of your library.
class GNEURAL_API CGNeural {
public:
    CGNeural(void);
    ~CGNeural(void); // Good practice to have a destructor

    // Delete the copy constructor and copy assignment operator
   // to prevent accidental copies of this resource-managing class.
    CGNeural(const CGNeural&) = delete;
    CGNeural& operator=(const CGNeural&) = delete;
    /// <summary>
    /// A helper function to easily create a network topology vector.
    /// </summary>
    Topology generateTopology(const int inputs, const int hidden, const int numHidden=1, const int outputs=1);

    /// <summary>
    /// Creates a new neural network, automatically selecting the best backend (GPU or CPU),
    /// and adds it to the internal list.
    /// </summary>
    /// <param name="topology">The network structure (e.g., {inputs, hidden, outputs}).</param>
    /// <returns>A raw pointer to the newly created network for immediate use. 
    /// The CGNeural class retains ownership of the object.</returns>
    InterfaceGNeuralNet* CreateAndAddNetwork(const Topology& topology, const std::string& file_name = "");

    /// <summary>
    /// Retrieves a pointer to a network at a specific index.
    /// </summary>
    /// <returns>A raw pointer to the network, or nullptr if the index is out of bounds.</returns>
    InterfaceGNeuralNet* GetNetwork(unsigned int index);




    /// <summary>
    /// Returns the number of networks currently being managed.
    /// </summary>
    unsigned int size(void) const { return m_networks.size(); };

private:
    // CRITICAL CHANGE: Store a vector of unique_ptrs to the INTERFACE.
    // This prevents object slicing and properly manages memory.
    std::vector<std::unique_ptr<InterfaceGNeuralNet>> m_networks;

};

// You can keep these C-style exports if you need them for other languages.
extern GNEURAL_API int nGNeural;
GNEURAL_API int fnGNeural(void);