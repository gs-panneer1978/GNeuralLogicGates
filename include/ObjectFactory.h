


// --- Include headers for all creatable types ---
#include "pch.h"
#include "constants.h"
#include "GNeuralConnection.h"
#include "GConnectionsList.h"
#include "GNeuronOpenCL.h"
#include "GNeuralNetOCL.h"
#include "GNeuron.h"
#include "GNeuralNet.h"

// #include "GNeuron.h" ... etc.

// --- STUB CLASSES FOR COMPILATION ---
// Create simple stubs for types you haven't defined yet.
//class GNeuron : public GObject { public: int GetTypeID() const override { return defNeuron; } };
//class GLayer : public GObject { public: int GetTypeID() const override { return defLayer; } };
// ... add other stubs as needed ...

// The Factory Function
std::unique_ptr<GObject> CreateObjectFromID(int typeID);

