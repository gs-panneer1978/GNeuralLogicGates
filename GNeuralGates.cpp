#pragma once
#include <iostream>
#include <vector>
#include <iomanip> // For std::fixed and std::setprecision
#include <cmath>   // For std::abs
#include <string>
#include <limits>
#include <stdexcept>


#include <memory>       // For std::unique_ptr
#include <sstream>      // For std::stringstream
#include <cstddef>      // For size_t
#include <functional> // NEW: for std::function
#include <map>        // NEW: for std::map
#include <filesystem> // NEW: for std::filesystem (C++17)

// --- FORWARD DECLARATIONS & Relative includes ---
// In your real project, you would #include your actual header files.
// Include the main header from your GNeural DLL library
#include "../GNeural/NetworkFactory.h"
#include "../GNeural/InterfaceGNeuralNet.h"
#include "../GNeural/GNeural.h"
#include "../GNeural/GNeuralNetOCL.h"
#include "../GNeural/GTypes.h"   // For types

// If you are using a compiler older than C++17, you might need an alternative
// for fileExists. See the helper function below.

using VectorDouble = std::vector<double>;
using Topology = std::vector<size_t>;

struct TrainingData {
    VectorDouble inputs;
    VectorDouble targets;
};

typedef enum {
    BUY = 1,
    SELL = -1,
    HOLD = 0
} ENUM_TRADE_ACTION;

/**
 * @brief Prints the contents of a vector of doubles to the console.
 * @param title A string title to print before the vector.
 * @param vec The vector of doubles to be printed.
 */
void printVectorToConsole(const std::string& title, const VectorDouble& vec) {
    std::cout << title << " [ ";
    for (const auto& val : vec) {
        std::cout << std::fixed << val << " ";
    }
    std::cout << "]" << std::endl;
}


/**
 * @brief Generates a random double within a specified range.
 * @param min The minimum value of the range.
 * @param max The maximum value of the range.
 * @return A random double between min and max.
 */
double random_double(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

/**
 * @brief Interprets a 3-element output vector to a string decision.
 * @param outputVec The output vector from the neural network, expected size 3.
 * @return A string: "SELL", "HOLD", "BUY", or "INVALID".
 */
std::string getDecision(const VectorDouble& outputVec) {
    if (outputVec[0] == 1.0) return "SELL";
    if (outputVec[1] == 1.0) return "HOLD";
    if (outputVec[2] == 1.0) return "BUY";
    return "INVALID";
}

// ===================================================================
//  THE MAIN APPLICATION CLASS
// ===================================================================

class NeuralNetworkTester {
public:
    /**
     * @brief Constructor for the NeuralNetworkTester class.
     * Initializes a map that associates string commands with their corresponding training functions.
     * This allows for easy extension and clean command dispatching.
     */
    NeuralNetworkTester() {
        // This map connects a string command (like "xor") to the function that trains it.
        // This is much cleaner than a big if-else block.
        m_trainingLaunchers["xor"] = [this]() { return this->runXorTraining(); };
        m_trainingLaunchers["and"] = [this]() { return this->runAndTraining(); };
        m_trainingLaunchers["or"] = [this]() { return this->runOrTraining(); };
        m_trainingLaunchers["nand"] = [this]() { return this->runNandTraining(); };
        m_trainingLaunchers["nor"] = [this]() { return this->runNorTraining(); };
        m_trainingLaunchers["xnor"] = [this]() { return this->runXnorTraining(); };
        m_trainingLaunchers["trade"] = [this]() { return this->runTradeTraining(); };
    }

    /**
     * @brief Runs the main application loop.
     * Displays a menu, reads user commands, and dispatches them to the appropriate handler functions.
     * The loop continues until the user enters 'q' or 'quit'.
     */
    void run() {
        displayMenu();
        std::string line;

        while (true) {
            std::cout << "\n> ";
            if (!std::getline(std::cin, line) || line == "q" || line == "quit") {
                std::cout << "Exiting program... Bye!" << std::endl;
                break;
            }

            std::stringstream ss(line);
            std::string command, argument;
            ss >> command >> argument; // Parse the line into a command and an argument

            // Convert to lowercase for easier matching
            for (char& c : command) { c = tolower(c); }
            for (char& c : argument) { c = tolower(c); }

            if (command.empty()) continue;

            if (command == "i" || command == "interactivetest") {
                handleInteractiveTestRequest(argument);
            }
            else {
                // Handle direct training commands (xor, and, etc.)
                auto it = m_trainingLaunchers.find(command);
                if (it != m_trainingLaunchers.end()) {
                    it->second(); // Call the training function from the map
                }
                else {
                    std::cout << "Invalid command. Please try again." << std::endl;
                    displayMenu();
                }
            }
        }
    }


private:
    // Member variables to hold the application's state
    CGNeural m_googiManager;
	// Pointer to the GNeuralNet network interface
    std::unique_ptr<InterfaceGNeuralNet> m_net;
	// Map to hold training functions
    std::map<std::string, std::function<bool()>> m_trainingLaunchers;

	ENUM_ACTIVATION m_activationFunction = ENUM_ACTIVATION::SIGMOID; // Default activation type

    /**
     * @brief Checks if a file exists on the filesystem.
     * Uses C++17 <filesystem> if available, otherwise falls back to a pre-C++17 method.
     * @param filename The path to the file to check.
     * @return True if the file exists, false otherwise.
     */
    bool fileExists(const std::string& filename) {
#if __cplusplus >= 201703L
        // C++17 or newer
        return std::filesystem::exists(filename);
#else
        // Pre-C++17 alternative:
        std::ifstream f(filename.c_str());
        return f.good();
#endif
    }

    /**
     * @brief Displays the main menu of available commands to the user.
     */
    void displayMenu() {
        std::cout << "========================================" << std::endl;
        std::cout << "  GNeuralLib (Logic Gates) Test Runner " << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Available Commands:" << std::endl;
        std::cout << "  - XOR       (Train the XOR gate) [Default] : LR(Eta): CPU=0.1 GPU=0.001, mom: 0.1" << std::endl;
        std::cout << "  - AND       (Train the AND gate) : LR(Eta): CPU=0.1 GPU=0.001, mom: 0.1" << std::endl; // NEW
        std::cout << "  - OR        (Train the OR gate) : LR(Eta): CPU=0.1 GPU=0.001, mom: 0.1" << std::endl;  // NEW
        std::cout << "  - NAND      (Train the NAND gate) : LR(Eta): CPU=0.1 GPU=0.001, mom: 0.1" << std::endl; // NEW
        std::cout << "  - NOR       (Train the NOR gate) : LR(Eta): CPU=0.1 GPU=0.001, mom: 0.1" << std::endl; // NEW
        std::cout << "  - XNOR      (Train the XNOR gate) : LR(Eta): CPU=0.1 GPU=0.001, mom: 0.1" << std::endl; // NEW
        std::cout << "  - TRADE     (Train the Trading Network : LR(Eta): CPU=0.1 GPU=0.001, mom: 0.1)" << std::endl; // NEW
        std::cout << "  - i <gate>            (e.g., 'i xor' to test the XOR gate)" << std::endl;
        std::cout << "  - q / quit  (Exit the program)" << std::endl;
    }

    /**
     * @brief Handles a request to run an interactive test for a specific gate.
     * It checks if a trained network file exists. If not, it prompts the user to train one on the fly.
     * @param gateName The name of the gate to test (e.g., "xor", "and").
     */
    void handleInteractiveTestRequest(std::string gateName) {
        if (gateName.empty()) {
            std::cout << "Please specify which gate to test (e.g., 'i xor'): ";
            std::getline(std::cin, gateName);
            if (gateName.empty()) return;
        }

        // Find the official training function for this gate to see if it's a valid gate
        auto it = m_trainingLaunchers.find(gateName);
        if (it == m_trainingLaunchers.end()) {
            std::cout << "Error: Unknown gate '" << gateName << "'. Cannot test." << std::endl;
            return;
        }

        // Standardize the name for the file (e.g., "XOR_Gate.nnw")
        std::string gateNameUpper = gateName;
        for (char& c : gateNameUpper) { c = toupper(c); }
        std::string filename = gateNameUpper + "_Gate.nnw";

        if (fileExists(filename)) {
            // If the file exists, run the test directly.
			std::cout << "Starting Interactive Test using CPU Neural schema file '" << filename << "' found." << std::endl;
            std::cout << "Loading trained network from '" << filename << "'..." << std::endl;

            runInteractiveTest(filename);
        } else if(fileExists("OCL_" + filename)) {
            // If the OpenCL version exists, run the test directly.
            std::cout << "Starting Interactive Test using OpenCL (GPU) Neural schema file 'OCL_" << filename << "' found." << std::endl;

            runInteractiveTest("OCL_" + filename);
		}
        else {
            // If the file does NOT exist, prompt the user to train it.
            std::cout << "Neural schema file '" << filename << "' not found." << std::endl;
            std::cout << "Would you like to train it now? (Y/n): ";
            std::string choice;
            std::getline(std::cin, choice);

            if (choice.empty() || choice[0] == 'y' || choice[0] == 'Y') {
                // Call the specific training function from our map.
                // The `it->second` is the std::function object itself.
                bool trainingWasSuccessful = it->second();

                // If training succeeded, proceed to the interactive test.
                if (trainingWasSuccessful) {
                    std::cout << "\nTraining complete. Proceeding to interactive test..." << std::endl;

                    runInteractiveTest(filename);
                }
                else {
                    std::cout << "\nTraining failed. Aborting interactive test." << std::endl;
                }
            }
            else {
                std::cout << "Aborting interactive test." << std::endl;
            }
        }
    }

    // ===================================================================
    // NEW: Generic Gate Training Engine
    // This single function replaces all the duplicated logic.
    // ===================================================================

    /**
     * @brief A generic engine for training a neural network for a logic gate.
     * This function handles topology creation, network training, and saving the result.
     * @param gateName The name of the gate (e.g., "XOR"), used for titling and filenames.
     * @param trainingSet The vector of training data (inputs and target outputs).
     * @param activationFunction The activation function to use for the network's neurons.
     * @return True if training was successful, false otherwise.
     */
    bool runGateTraining(const std::string& gateName, const std::vector<TrainingData>& trainingSet, ENUM_ACTIVATION activationFunction = SIGMOID) {
        std::cout << "\n--- GNeural Library: " << gateName << " Gate Training Test ---" << std::endl;

        // The title for the save file (e.g., "AND_Gate.nnw")
        std::string title = gateName + "_Gate";

        // Get the desired network structure from the user
        Topology topology = getTopologyFromUser();

        // Create the network
        m_net = NetworkFactory::CreateNewNetwork(topology);
        if (!m_net) {
            std::cerr << "FATAL ERROR: Could not create neural network!" << std::endl;
            return false;
        }

        m_net->SetActivationType(activationFunction);
        // Train the network with the provided data
        bool trainingComplete = trainNetwork(trainingSet);

        // Save and verify if training was successful
        if (trainingComplete) {
            verifyAndSaveNetwork(title, trainingSet);
        }
        else {
            std::cout << "\n--- Training Failed ---" << std::endl;
            std::cout << "Network did not converge within the maximum number of passes. Try again after increasing the number of passes." << std::endl;
        }

        std::cout << "\nTest complete. Ready for new command." << std::endl;
		return trainingComplete; // Return true if training was successful
    }

    // ===================================================================
    // NEW: Specific Gate Launcher Methods
    // These are now very simple. They just define the data and call the engine.
    // ===================================================================

    /**
     * @brief Prepares the training data for the XOR gate and initiates training.
     * @return The result of the training process (true for success, false for failure).
     */
    bool runXorTraining() {
        const std::vector<TrainingData> trainingSet = {
            {{0.0, 0.0}, {0.0}},
            {{0.0, 1.0}, {1.0}},
            {{1.0, 0.0}, {1.0}},
            {{1.0, 1.0}, {0.0}}
        };
        return runGateTraining("XOR", trainingSet);
    }

    /**
     * @brief Prepares the training data for the AND gate and initiates training.
     * @return The result of the training process (true for success, false for failure).
     */
    bool runAndTraining() {
        const std::vector<TrainingData> trainingSet = {
            {{0.0, 0.0}, {0.0}},
            {{0.0, 1.0}, {0.0}},
            {{1.0, 0.0}, {0.0}},
            {{1.0, 1.0}, {1.0}}
        };
        return runGateTraining("AND", trainingSet);
    }

    /**
     * @brief Prepares the training data for the OR gate and initiates training.
     * @return The result of the training process (true for success, false for failure).
     */
    bool runOrTraining() {
        const std::vector<TrainingData> trainingSet = {
            {{0.0, 0.0}, {0.0}},
            {{0.0, 1.0}, {1.0}},
            {{1.0, 0.0}, {1.0}},
            {{1.0, 1.0}, {1.0}}
        };
        return runGateTraining("OR", trainingSet);
    }

    /**
     * @brief Prepares the training data for the NAND gate and initiates training.
     * @return The result of the training process (true for success, false for failure).
     */
    bool runNandTraining() {
        const std::vector<TrainingData> trainingSet = {
            {{0.0, 0.0}, {1.0}},
            {{0.0, 1.0}, {1.0}},
            {{1.0, 0.0}, {1.0}},
            {{1.0, 1.0}, {0.0}}
        };
        return runGateTraining("NAND", trainingSet);
    }


    /**
     * @brief Prepares the training data for the NOR gate and initiates training.
     * @return The result of the training process (true for success, false for failure).
     */
    bool runNorTraining() {
        const std::vector<TrainingData> trainingSet = {
            {{0.0, 0.0}, {1.0}},
            {{0.0, 1.0}, {0.0}},
            {{1.0, 0.0}, {0.0}},
            {{1.0, 1.0}, {0.0}}
        };
        return runGateTraining("NOR", trainingSet);
    }

    /**
     * @brief Prepares the training data for the XNOR gate and initiates training.
     * @return The result of the training process (true for success, false for failure).
     */
    bool runXnorTraining() {
        const std::vector<TrainingData> trainingSet = {
            {{0.0, 0.0}, {1.0}},
            {{0.0, 1.0}, {0.0}},
            {{1.0, 0.0}, {0.0}},
            {{1.0, 1.0}, {1.0}}
        };
        return runGateTraining("XNOR", trainingSet);
    }

	/**
     * @brief Prepares a sample training dataset for a trading decision model and initiates training.
     * This model uses one-hot encoding for the output: {SELL, HOLD, BUY}.
     * It uses the SIGMOID activation function.
     * @return The result of the training process (true for success, false for failure).
     */
    bool runTradeTraining() {
        // for complex decision making of for a more robust and standard approach for 3 or more distinct classes
        // is to use one-hot encoding for the output.
        // 3 input neurons, a large hidden layer and 3 output neurons
        // Activation: For this setup, you can go back to using the Sigmoid function on all output neurons.
        // When you feed forward an input, the network's output might look like {0.85, 0.12, 0.05},
        // and you would interpret this as a "Sell" decision because the first neuron has the highest value.
        m_activationFunction = ENUM_ACTIVATION::SIGMOID; // Set the activation function to SIGMOID

        const std::vector<TrainingData> trainingSet = {
            // Each entry:
            // {{open,  close,  low,   high,   atr,    cci,     macd,   psar}, {SELL, HOLD, BUY}}

            // --- BUY SIGNALS ---
            // Strong bullish candle, positive indicators
            {{1.1205, 1.1285, 1.1200, 1.1290, 0.0090,  110.5,  0.0018, 1.1195}, {0.0, 0.0, 1.0}},
            {{1.0850, 1.0955, 1.0845, 1.0960, 0.0110,  125.0,  0.0025, 1.0840}, {0.0, 0.0, 1.0}},
            {{1.1310, 1.1390, 1.1305, 1.1400, 0.0095,   95.8,  0.0015, 1.1298}, {0.0, 0.0, 1.0}},
            {{1.0920, 1.1010, 1.0915, 1.1018, 0.0098,  130.2,  0.0021, 1.0905}, {0.0, 0.0, 1.0}},
            {{1.1150, 1.1220, 1.1145, 1.1228, 0.0078,   88.0,  0.0012, 1.1138}, {0.0, 0.0, 1.0}},
            {{1.0760, 1.0840, 1.0755, 1.0845, 0.0085,  105.3,  0.0019, 1.0750}, {0.0, 0.0, 1.0}},
            {{1.1400, 1.1495, 1.1398, 1.1500, 0.0105,  140.0,  0.0028, 1.1390}, {0.0, 0.0, 1.0}},

            // --- SELL SIGNALS ---
            // Strong bearish candle, negative indicators
            {{1.1350, 1.1270, 1.1265, 1.1355, 0.0085, -115.3, -0.0016, 1.1360}, {1.0, 0.0, 0.0}},
            {{1.1000, 1.0910, 1.0905, 1.1005, 0.0090, -132.0, -0.0022, 1.1010}, {1.0, 0.0, 0.0}},
            {{1.0890, 1.0800, 1.0795, 1.0895, 0.0100, -128.5, -0.0028, 1.0905}, {1.0, 0.0, 0.0}},
            {{1.1450, 1.1365, 1.1360, 1.1455, 0.0095, -105.0, -0.0019, 1.1460}, {1.0, 0.0, 0.0}},
            {{1.1280, 1.1200, 1.1198, 1.1285, 0.0088, -140.1, -0.0024, 1.1290}, {1.0, 0.0, 0.0}},
            {{1.0950, 1.0870, 1.0865, 1.0955, 0.0080, -112.7, -0.0015, 1.0965}, {1.0, 0.0, 0.0}},
            {{1.1110, 1.1020, 1.1015, 1.1115, 0.0092, -135.8, -0.0026, 1.1125}, {1.0, 0.0, 0.0}},

            // --- HOLD SIGNALS ---
            // Indecisive candle (doji/spinning top), neutral indicators
            {{1.1300, 1.1305, 1.1280, 1.1325, 0.0045,   10.2,  0.0001, 1.1275}, {0.0, 1.0, 0.0}},
            {{1.0910, 1.0908, 1.0890, 1.0930, 0.0040,   -5.8, -0.0002, 1.0885}, {0.0, 1.0, 0.0}},
            {{1.1250, 1.1255, 1.1235, 1.1270, 0.0035,   15.5,  0.0003, 1.1230}, {0.0, 1.0, 0.0}},
            {{1.0880, 1.0875, 1.0860, 1.0895, 0.0038,  -20.0, -0.0004, 1.0900}, {0.0, 1.0, 0.0}},
            {{1.1180, 1.1183, 1.1165, 1.1195, 0.0030,    2.5,  0.0000, 1.1160}, {0.0, 1.0, 0.0}},
            {{1.1330, 1.1332, 1.1310, 1.1350, 0.0042,   25.1,  0.0005, 1.1305}, {0.0, 1.0, 0.0}},
            {{1.0980, 1.0978, 1.0960, 1.0995, 0.0039,  -12.3, -0.0001, 1.0955}, {0.0, 1.0, 0.0}},
        };
        // We are using the Sigmoid function on all output neurons instead of TanH.
        // When you feed forward an input, the network's output might look like {0.85, 0.12, 0.05},
        // and you would interpret this as a "Sell" decision because the first neuron has the highest value.
		// if necessary, add a - sign to the first neuron output to indicate SELL.
        return runGateTraining("Trade", trainingSet, SIGMOID);
	}


    /**
     * @brief Runs an interactive testing session for a pre-trained network.
     * This function is data-driven. It loads a network from a file, determines the required number
     * of inputs and outputs from the loaded topology, and then enters a loop to accept user inputs,
     * perform inference, and display the results in an interpreted format.
     * @param network_file The path to the saved neural network file (.nnw).
     */
    void runInteractiveTest( const std::string& network_file) {
        std::cout << "\n--- Interactive Inference Test ---" << std::endl;

        // 1. Create a network object and load the state from the file.
        std::cout << "Loading trained network from '" << network_file << "'..." << std::endl;
        std::unique_ptr<InterfaceGNeuralNet> loadedNet = NetworkFactory::LoadNetworkFromFile(network_file);

        if (!loadedNet) {
            std::cerr << "Failed to load the network. Aborting test." << std::endl;
            return;
        }

        // 2. Get the definitive topology FROM the loaded network.
        Topology topology = loadedNet->getTopology();
        if (topology.size() < 2) {
            std::cerr << "Invalid topology loaded from file. Aborting." << std::endl;
            return;
        }
		double final_result = 0.0; // Variable to hold the final result
        const size_t numInputs = topology.front();
        const size_t numOutputs = topology.back();

        std::cout << "Network '" << network_file << "' loaded successfully." << std::endl;
        std::cout << "--> Required Inputs: " << numInputs << ", Outputs: " << numOutputs << std::endl;
        std::cout << "Type 'q' or 'quit' to exit." << std::endl;

        std::string line;
        while (true) {
            // 3. Prompt the user with dynamic information.
            std::cout << "\nEnter " << numInputs << " numbers separated by spaces > ";
            if (!std::getline(std::cin, line) || line == "q" || line == "quit") {
                break;
            }

            std::stringstream ss(line);
            VectorDouble inputs;
            inputs.reserve(numInputs); // Good practice for performance
            double value;

            // 4. Dynamically read the required number of inputs.
            for (size_t i = 0; i < numInputs; ++i) {
                if (!(ss >> value)) {
                    // Break if we can't read a value (e.g., not enough inputs provided)
                    break;
                }
                inputs.push_back(value);
            }

            // 5. Validate the input.
            if (inputs.size() != numInputs) {
                std::cerr << "Error: Invalid input. Expected " << numInputs << " numbers." << std::endl;
                continue;
            }

            // 6. Perform inference.
            loadedNet->feedForward(inputs);

            VectorDouble results;
            loadedNet->getResults(results);

            // 7. Display results based on the number of output neurons.
            std::cout << "  Raw Output Vector: [";
            for (size_t i = 0; i < results.size(); ++i) {
                std::cout << std::fixed << std::setprecision(4) << results[i] << (i == results.size() - 1 ? "" : ", ");
            }
            std::cout << "]" << std::endl;

            if (numOutputs == 1) {
                // Case 1: Single output neuron (like for AND, XOR gates or basic regression)
                double raw_output = results[0];
                // NOTE: The threshold depends on the activation function!
                // 0.5 for Sigmoid [0, 1], 0.0 for Tanh [-1, 1]
                double threshold = 0.0; // Assuming Tanh for your trading model
                if (getGateName(network_file) != "Trade") threshold = 0.5; // Or check activation type

                std::cout << "  Interpreted Result: " << (raw_output > threshold ? 1 : 0) << std::endl;

            }
            else {
                // Case 2: Multiple output neurons (like for Buy/Sell/Hold classification)
                auto max_it = std::max_element(results.begin(), results.end());
                int winning_index = std::distance(results.begin(), max_it);


                std::cout << "  Interpreted Result: ";
                std::cout << "Class " << winning_index << " with value: " << std::fixed << std::setprecision(4) << *max_it << " which signifies : ";


				final_result = 0.0; // Reset final result for each iteration

                switch (winning_index) {
                    case 0: // Assuming index 0 corresponds to SELL
                        final_result = -1 * results.at(0);
                        std::cout <<  EnumToString(rawToSignal(final_result)) << "[" << final_result << "]" << std::endl;
                        break;
                    case 1: // Assuming index 1 corresponds to HOLD
                        std::cout << EnumToString(rawToSignal(final_result)) << "[" << final_result << "]" << std::endl;
                        final_result = results.at(1);
                        break;
                    case 2: // Assuming index 2 corresponds to BUY
                        std::cout << EnumToString(rawToSignal(final_result)) << "[" << final_result << "]"  << std::endl;
                        final_result = results.at(2);
                        break;
                    default:
                        std::cout << "Undefined Class (" << winning_index << ")" << std::endl;
                        break;
                    }
            }
        }

        std::cout << "\nExiting interactive test." << std::endl;
    }

    /**
     * @brief Converts a raw numerical output from the trading model to a trade action.
     * @param raw_output The raw output value from a neuron, passed by reference.
     * @return An ENUM_TRADE_ACTION (BUY, SELL, or HOLD).
     */
    ENUM_TRADE_ACTION rawToSignal(double &raw_output) {
		std::cout << "  Raw Output: " << std::fixed << std::setprecision(4) << raw_output << std::endl;
        if (raw_output > 0.1) {
            return BUY; // Buy signal
        }
        else if (raw_output < -0.1) {
            return SELL; // Sell signal
        }
        else {
            return HOLD; // Hold signal
        }
	}

    /**
     * @brief Converts an ENUM_TRADE_ACTION to its string representation.
     * @param tradeAction The enum value to convert.
     * @return A string ("BUY", "SELL", or "HOLD").
     */
    std::string EnumToString(ENUM_TRADE_ACTION tradeAction) {
        switch (tradeAction) {
        case 1: return "BUY";
        case -1: return "SELL";
        default: return "HOLD";
        }
    }

    /**
     * @brief Manages the core training loop for the neural network.
     * It iterates through the training data, performing feed-forward and back-propagation passes
     * until the network's error is low enough for a set number of consecutive epochs or the max number of passes is reached.
     * @param trainingSet The dataset to train on.
     * @return True if the network trained successfully (converged), false otherwise.
     */
    bool trainNetwork(const std::vector<TrainingData>& trainingSet) {
        // Training parameters
        const double marginOfError = 0.1;
        const int maxPasses = 500000;
        const int requiredSuccesses = 3;
        int consecutiveSuccesses = 3;

        std::cout << "\nStarting training...\n";
        for (int pass = 1; pass <= maxPasses; ++pass) {
            bool epochWasSuccessful = true;
            double epochError = 0.0;

            for (const auto& data : trainingSet) {

                m_net->feedForward(data.inputs);
                m_net->backPropagate(data.targets);

                VectorDouble results;
                m_net->getResults(results);

                double error = std::abs(results[0] - data.targets[0]);
                epochError += error;

                if (error > marginOfError) {
                    epochWasSuccessful = false;
                }
            }

            consecutiveSuccesses = epochWasSuccessful ? consecutiveSuccesses + 1 : 0;

            if (pass % 100 == 0) {
                std::cout << "Pass " << std::setw(5) << pass << " | "
                    << "Consecutive Successes: " << std::setw(2) << consecutiveSuccesses << "/" << requiredSuccesses
                    << " | Avg Error: " << std::fixed << std::setprecision(4) << (epochError / trainingSet.size())
                    << std::endl;
                m_net->Display("GNeuralNet : Pass" + std::to_string(pass));
            }

            if (consecutiveSuccesses >= requiredSuccesses) {
                std::cout << "\n--- Training Successful! ---" << std::endl;
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Prompts the user to define the network topology.
     * Asks for the number of input neurons, hidden layers, neurons per hidden layer, and output neurons.
     * @return A Topology (vector of size_t) representing the network structure.
     */
    Topology getTopologyFromUser() {
        Topology topology;
        std::string line;
        size_t num_inputs, num_hidden, num_hidden_layers, num_outputs;

        std::cout << "How many input neurons? (Default: 2): ";
        std::getline(std::cin, line);
        num_inputs = line.empty() ? 2 : std::stoi(line);

        std::cout << "How many hidden layers? (Default: 1): ";
        std::getline(std::cin, line);
        num_hidden_layers = line.empty() ? 1 : std::stoi(line);

        std::cout << "How many neurons per hidden layer? (Default: 3): ";
        std::getline(std::cin, line);
        num_hidden = line.empty() ? 3 : std::stoi(line);

        std::cout << "How many output neurons? (Default: 1): ";
        std::getline(std::cin, line);
        num_outputs = line.empty() ? 1 : std::stoi(line);

        topology.push_back(num_inputs);
        for (size_t i = 0; i < num_hidden_layers; ++i) {
            topology.push_back(num_hidden);
        }
        topology.push_back(num_outputs);

        return topology;
    }

    /**
     * @brief Saves the trained network to a file and performs a final verification pass.
     * It prints the network's output for each entry in the training set to show how well it learned.
     * @param title The base name for the output file (e.g., "XOR_Gate").
     * @param trainingSet The original training data used for verification.
     */
    void verifyAndSaveNetwork(const std::string& title, const std::vector<TrainingData>& trainingSet) {
        std::string filename = title + ".nnw";
        std::cout << "Saving trained network to '" << filename << "'..." << std::endl;

        if (m_net->saveNetwork(filename)) {
            std::cout << "Network saved successfully." << std::endl;
        }
        else {
            std::cerr << "Error: Failed to save the network." << std::endl;
        }

        std::cout << "\n--- Final Verification ---" << std::endl;
        for (const auto& data : trainingSet) {
            m_net->feedForward(data.inputs);
            VectorDouble finalResults;
            m_net->getResults(finalResults);

            std::cout << "Input:  [";
            for (size_t i = 0; i < data.inputs.size(); ++i) {
                std::cout << std::fixed << std::setprecision(4) << data.inputs[i] << (i == data.inputs.size() - 1 ? "" : ", ");
            }
            std::cout << "] (Target: [";
            for (size_t i = 0; i < data.targets.size(); ++i) {
                std::cout << static_cast<int>(data.targets[i]) << (i == data.targets.size() - 1 ? "" : ", ");
            }
            std::cout << "] -> Output: [";
            for (size_t i = 0; i < finalResults.size(); ++i) {
                std::cout << std::fixed << std::setprecision(4) << finalResults[i] << (i == finalResults.size() - 1 ? "" : ", ");
            }
            std::cout << "])" << std::endl;
        }
    }

    /**
     * @brief Extracts a base name from a full filename string.
     * For example, it extracts "XOR" from "XOR_Gate.nnw".
     * @param full_name The full filename string.
     * @return The extracted base name.
     */
    std::string getGateName(const std::string& full_name) {
        if (full_name.empty()) {
            return " "; // Return empty string if input is empty.
        }

        // Find the position of the first underscore.
        size_t underscore_pos = full_name.find('_');

        // If an underscore is found...
        if (underscore_pos != std::string::npos) {
            // ...return the part of the string from the beginning up to the underscore.
            return full_name.substr(0, underscore_pos);
        }

        // --- Optional: Handle cases with no underscore ---
        // If no underscore is found, maybe the gate name is terminated by a dot.
        size_t dot_pos = full_name.find('.');
        if (dot_pos != std::string::npos) {
            return full_name.substr(0, dot_pos);
        }

        // If no underscore or dot is found, return the whole string.
        return full_name;
    }


};

/**
 * @brief The main entry point of the program.
 * Creates an instance of the NeuralNetworkTester application and runs it.
 * @return 0 on successful execution.
 */
int main() {
    // Create an instance of our application class
    NeuralNetworkTester app;

    // Run the application
    app.run();

    return 0;
}
