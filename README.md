# GNeural Logic Gate Trainer
 Logic gates running on a MLP (FeedForward/Backprop) neural network that runs on both CPU & GPU. GNeural LogicGates  is a comprehensive and creative project which demonstrates the complex & deterministic nature of Neural Networks. 

![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows-informational)
![Visual Studio](https://img.shields.io/badge/Visual%20Studio-2022-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)

A powerful C++ console application for training and testing neural networks on classic logic gates (XOR, AND, OR, etc.), built upon the core `GNeural` library. This project serves as a practical demonstration of the core functionalities of `GNeural`, including network creation, training via backpropagation, serialization (saving/loading), and interactive testing.

---

## Features

- **Multi-Gate Training:** Train networks for all standard logic gates:
  - `XOR`, `AND`, `OR`, `NAND`, `NOR`, `XNOR`
- **Dynamic Topology:** Easily define custom network architectures (number of layers and neurons) at runtime.
- **Interactive Console UI:** A user-friendly command-line interface to select gates, train networks, and run tests.
- **Serialization:** Save fully trained network weights and topology to a `.nnw` file for later use.
- **Train-on-Demand:** If you try to test a gate that hasn't been trained yet, the application will prompt you to train it on the fly.
- **Interactive Inference:** Load any saved `.nnw` file and test its predictions with your own custom inputs.
- **Cross-Platform Potential:** Built with standard C++, demonstrating how `GNeural` can be used in various environments.

## 🛠️ Getting Started

### Prerequisites

- **Visual Studio 2019 or newer** (with the "Desktop development with C++" workload)
- **C++17** compliant compiler
- The **`GNeural`** library (This project assumes `GNeural.dll` and its headers are available).

### Setup & Compilation

This project is designed to work as a consumer of the `GNeural` library. The recommended setup is to place the `GNeural` library folder adjacent to this project's folder.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/GNeuralLogicGates.git
    cd GNeuralLogicGates
    ```

2.  **Set up the `GNeural` Dependency:**
    Ensure the compiled `GNeural.dll`, `GNeural.lib`, and header files are accessible. The project is pre-configured to look for them in a parallel directory structure like this:
    ```
    /your_dev_folder/
    ├── include/          <-- The GNeural library project
    └── GNeuralLogicGates/  <-- This project
    ```
    If your structure is different, you will need to update the project properties in Visual Studio:
    - **Additional Include Directories:** `C/C++ -> General`
    - **Additional Library Directories:** `Linker -> General`

3.  **Open the Solution in Visual Studio:**
    Open the `.sln` file in Visual Studio.

4.  **Build the Project:**
    - Select a build configuration (e.g., `x64 Debug` or `x64 Release`).
    - Build the solution (`Build -> Build Solution` or `F7`). A Post-Build event will automatically copy the required `GNeural.dll` to the output directory.

## 🕹️ How to Use

Run the compiled executable (`GNeuralGates.exe`) from the command line or directly from Visual Studio. You will be greeted with a menu of available commands.

### Training a Network

To train a network, simply type the name of the gate you want to train.

```
> xor
```

The application will then prompt you for the network topology and training parameters:

```
--- GNeural Library: XOR Gate Training Test ---
How many input neurons? (Default: 2):
How many hidden layers? (Default: 1):
How many neurons per hidden layer? (Default: 3):
How many output neurons? (Default: 1):
Enter learning rate and momentum (default: 0.01, 0.5):

Starting training...
Pass   100 | Consecutive Successes:  0/3 | Avg Error: 0.2481
Pass   200 | Consecutive Successes:  0/3 | Avg Error: 0.2350
...
Pass  1500 | Consecutive Successes:  3/3 | Avg Error: 0.0098

--- Training Successful! ---
Saving trained network to 'XOR_Gate.nnw'...
```

### Interactive Testing

The most powerful feature is the interactive test mode. You can test any gate that has already been trained and saved.

1.  **Start the interactive test:**
    ```
    > i xor
    ```
    If `XOR_Gate.nnw` does not exist, it will prompt you to train it first.

2.  **Provide Inputs:**
    Once the network is loaded, you can provide inputs to see the network's prediction.
    ```
    --- Interactive Inference Test ---
    Network 'XOR_Gate.nnw' loaded successfully.
    --> Required Inputs: 2, Outputs: 1
    Type 'q' or 'quit' to exit.

    Enter 2 numbers separated by spaces > 1 0
      Raw Output Vector: [0.9854]
      Interpreted Result: 1

    Enter 2 numbers separated by spaces > 1 1
      Raw Output Vector: [0.0112]
      Interpreted Result: 0
    ```

### Available Commands

| Command | Description |
| :--- | :--- |
| `xor` | Train the XOR gate network. |
| `and` | Train the AND gate network. |
| `or` | Train the OR gate network. |
| `nand` | Train the NAND gate network. |
| `nor` | Train the NOR gate network. |
| `xnor` | Train the XNOR gate network. |
| `i <gate>` | Run an interactive test for a saved gate (e.g., `i and`). |
| `q` / `quit` | Exit the application. |


## Training and Testing with GPU Acceleration

This block explains how to use a GPU-accelerated version of the library and shows what the output might look like.


### 🚀 GPU-Accelerated Training (OpenCL)

This application can also leverage the power of your GPU for significantly faster training passes, especially on larger networks. This functionality is enabled by the `GNeuralNetOCL` backend within the `GNeural` library.

#### Usage

To use the GPU, you would typically select it at runtime or have a version of the application specifically compiled to use the OpenCL backend. When a GPU-enabled network is trained, the output will indicate that the OpenCL device is being used.


**Example Training Session:**
```

> trade_gpu

--- GNeural Library: TRADE_GPU Training Test ---
Selected OpenCL Platform: NVIDIA CUDA
Selected OpenCL Device: NVIDIA GeForce RTX 3080

How many input neurons? (Default: 2): 8
How many hidden layers? (Default: 1): 10
How many neurons per hidden layer? (Default: 3): 100
How many output neurons? (Default: 1): 3
Network Learning Rate (0.01): 0.01
Momentum (0.1): 0.1

GNeuralNet default constructor called (empty object created).
GNeuralNet Constructor called with topology size: 12 Or filename:
Topology:  [ 8 100 100 100 100 100 100 100 100 100 100 3 ]
GNeuron created with 100 outputs, activationType: 0 and index 0
GNeuron created with 100 outputs, activationType: 0 and index 1
GNeuron created with 100 outputs, activationType: 0 and index 2
Activation Function: 0
GNeuron created with 100 outputs, activationType: 0 and index 3
...
..
GNeuralNet::build => Created 12 layers with topology:
==================================================
  NETWORK STATE: GNeuralNet(LR : 0.100000               Momentum: 0.500000 )
==================================================
Layer 0 (Input Layer) - Neurons: 8 + 1 Bias
--------------------------------------------------
  Neuron    0: Output Val:   0.0000 | Gradient:   0.0000
  Neuron    1: Output Val:   0.0000 | Gradient:   0.0000
  Neuron    2: Output Val:   0.0000 | Gradient:   0.0000
  Neuron    3: Output Val:   0.0000 | Gradient:   0.0000
  Neuron    4: Output Val:   0.0000 | Gradient:   0.0000
  Neuron    5: Output Val:   0.0000 | Gradient:   0.0000
  Neuron    6: Output Val:   0.0000 | Gradient:   0.0000
  Neuron    7: Output Val:   0.0000 | Gradient:   0.0000
  Bias Neuron 8: Output Val:   1.0000 | Gradient:   0.0000

Layer 1 (Hidden Layer) - Neurons: 100 + 1 Bias
--------------------------------------------------
  Neuron    0: Output Val:   0.0000 | Gradient:   0.0000
...
...
...
...
  Neuron   92: Output Val:   0.0000 | Gradient:   0.0000
  Neuron   93: Output Val:   0.0000 | Gradient:   0.0000
  Neuron   94: Output Val:   0.0000 | Gradient:   0.0000
  Neuron   95: Output Val:   0.0000 | Gradient:   0.0000
  Neuron   96: Output Val:   0.0000 | Gradient:   0.0000
  Neuron   97: Output Val:   0.0000 | Gradient:   0.0000
  Neuron   98: Output Val:   0.0000 | Gradient:   0.0000
  Neuron   99: Output Val:   0.0000 | Gradient:   0.0000
  Bias Neuron 100: Output Val:   1.0000 | Gradient:   0.0000

Layer 11 (Output Layer) - Neurons: 3 + 1 Bias
--------------------------------------------------
  Neuron    0: Output Val:   0.0000 | Gradient:   0.0000
  Neuron    1: Output Val:   0.0000 | Gradient:   0.0000
  Neuron    2: Output Val:   0.0000 | Gradient:   0.0000
  Bias Neuron 3: Output Val:   1.0000 | Gradient:   0.0000

==================================================
OpenCL GPU detected. Would you prefer creating new GPU-accelerated network?Please choose a network backend:
  1. CPU (GNeuralNet)
  2. GPU (GNeuralNetOCL) - Recommended
Your choice: 2

Creating GPU-accelerated network...

==================================================
 OpenCL Network Status
==================================================

Learning Rate: 0.0100
Momentum: 0.5000

Using fp64 ?:0 

..
..

--- Layer with 100 neurons ---
Gradients (100 values):  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
Weights (10100 values):    -0.3236 -0.0307 -0.0137 0.1569  0.0388  -0.2617 -0.4446 0.0621  0.3478  -0.3534 0.1352  -0.1105-0.0289  -0.4754 0.1408  0.2422  -0.1896 0.0890  0.0659  -0.2806 -0.5276 -0.1954 0.2596  -0.0049 -0.1214 0.0855  -0.33420.5742   0.1460  -0.0330 -0.5530 -0.2643 -0.3794 -0.1554 -0.1231 -0.2077 -0.1524 0.4199  0.0222  0.0385  -0.4234 0.4339 0.2271   0.4599  0.0725  -0.4827 0.5375  -0.2412 -0.4314 -0.3709 -0.0481 -0.3011 -0.0484 -0.1173 0.4872  -0.5254 0.1334 -0.0832  0.4019  -0.3219 0.5448  -0.0229 -0.0136 0.0544  0.1486  0.3621  -0.1349 0.1665  -0.1404 -0.3476 -0.3295 0.4015 0.2344   -0.2241 -0.2745 0.1837  -0.5695 -0.5122 0.5441  -0.2367 -0.0193 0.0577  0.5203  -0.4148 -0.5499 -0.3677 0.5153 -0.0435  0.4119  -0.2860 ...
Outputs (100 values):    -0.2077 -0.1524 0.4199  0.0000  0.01020  0.360`  0.5375  -0.2412 -0.4314  ...
```
#### The Output Layer 
```
--- Layer with 3 neurons ---
Gradients (3 values):   -0.0073 0.0425  -0.0434
Weights (303 values):   -0.2608 0.1765  0.2423  0.0250  -0.3709 -0.1334 0.1005  -0.1281 -0.0319 -0.0084 0.0899  0.1491  -0.1562 -0.2383 -0.2150 0.0457  -0.3904-0.2373 0.1148  -0.3503 0.2029  0.1910  0.1371  0.0707  -0.2411 -0.0647 0.0580  -0.1989 -0.0097 -0.0816 -0.2147 0.2084  0.1033  0.3286  -0.2459 0.1464  -0.05530.0788  0.1390  -0.0814 -0.1236 0.1749  0.0315  0.2303  -0.0723 0.0824  -0.1747 0.1298  0.1401  0.0013  0.2322  0.0070  0.0489  -0.2420 0.0038  0.2151  -0.3195-0.1031 -0.2199 -0.1811 -0.2358 0.2257  -0.0821 -0.2266 0.1963  -0.1340 0.2809  -0.4646 -0.0680 -0.2213 -0.0019 -0.2012 0.0298  -0.1294 0.0756  -0.0919 0.0125-0.1611  -0.1091 0.0182  0.1003  0.0198  0.1766  0.1312  -0.0689 0.1317  0.2418  0.2137  0.2325  -0.3372 0.0181  -0.1478 0.3461  0.0257  -0.2249 -0.0166 -0.2184-0.1750 -0.1513 -0.0769 -0.5539 0.0922  -0.0806 0.1536  -0.4830 0.3511  -0.2901 0.0551  0.1008  0.0351  0.1755  0.0653  -0.1918 0.0640  0.0532  -0.2234 -0.1248-0.0625 -0.1549 0.0860  0.1551  0.1963  -0.0966 -0.0664 0.0526  0.3152  0.1407  0.2114  0.1634  0.1629  -0.0347 -0.2273 -0.1243 0.0287  -0.3304 -0.0502 -0.20200.2332  -0.0619 -0.1386 0.1498  -0.2285 -0.0386 -0.0060 0.1485  -0.2006 -0.0040 -0.1089 -0.0626 -0.1802 -0.1293 -0.3661 -0.1424 0.0977  0.0835  -0.2472 0.02010.3839   -0.0005 0.1658  -0.2579 -0.1885 -0.2232 0.1892  -0.0072 -0.1709 -0.1374 -0.0304 0.0438  -0.0416 -0.1523 0.0563  0.1864  -0.2056 -0.3220 0.0888  -0.2221-0.3337 -0.0188 0.0298  0.1669  -0.0388 0.0024  -0.0868 -0.2159 -0.1275 0.1879  0.1231  0.1348  -0.1994 0.0626  0.1749  -0.0502 -0.0249 0.1499  0.0941  -0.3935-0.1649 -0.0248 -0.1341 -0.2623 0.8880  -0.2771 0.0255  -0.1426 0.4433  -0.3734 0.2427  0.1434  -0.2415 -0.1128 -0.2243 -0.2177 -0.0782 0.0090  -0.1029 0.21650.1536   0.2866  0.4592  -0.2216 -0.1215 -0.0209 0.0366  -0.3129 -0.0734 -0.2218 0.1231  0.2318  0.2291  0.0576  -0.2070 0.1307  0.2251  0.0807  -0.2349 -0.2158-0.1915 -0.0619 -0.3082 0.1857  -0.1884 0.1782  -0.1513 0.1311  -0.2122 0.2099  -0.0042 0.0162  0.1062  0.1246  0.1272  0.3450  0.1174  0.0952  0.0880  -0.01040.0443  -0.2882 -0.1402 -0.0382 0.3992  -0.0128 0.2015  -0.2079 0.1757  -0.3702 -0.2016 -0.1941 0.3511  0.0495  0.0648  0.0767  0.0098  0.1695  -0.0336 -0.04450.0813  -0.0519 -0.2503 0.1757  -0.3757 0.0424  0.3603  0.1803  0.1868  0.4642  -0.0645 -0.4383 0.2150  -0.0152 -0.1524 -0.1084 0.0611  -0.4368 -0.1404 0.0989-0.0837  0.0252  -0.1789 -0.1871 0.1469  -0.7441
Outputs (3 values):     0.0895  0.7642  0.2388

--- Training Successful! ---
Saving trained network to 'OCL_TRADE_Gate.nnw'...

--- Final Verification ---
Input:  [1.1205, 1.1285, 1.1200, 1.1290, 0.0090, 110.5000, 0.0018, 1.1195] -> Output: [0.0012, 0.0345, 0.9881] (Target: [0, 0, 1])
Input:  [1.1350, 1.1270, 1.1265, 1.1355, 0.0085, -115.3000, -0.0016, 1.1360] -> Output: [0.9902, 0.0211, 0.0005] (Target: [1, 0, 0])
Input:  [1.1300, 1.1305, 1.1280, 1.1325, 0.0045, 10.2000, 0.0001, 1.1275] -> Output: [0.0450, 0.9753, 0.0312] (Target: [0, 1, 0])
```

*Notice how the output confirms the use of an OpenCL device and successfully classifies the BUY, SELL, and HOLD signals.*


### Block 2: Trading Signal Prediction

### 📈 Advanced Example: Trading Signal Prediction

Beyond simple logic gates, this project includes a more complex example for predicting financial trading signals (BUY, SELL, or HOLD). This demonstrates the library's ability to handle multi-input, multi-output classification problems.

#### Network Architecture

-   **Inputs (8 neurons):** Simulates real market data points such as `open`, `close`, `low`, `high`, and various technical indicators (`ATR`, `CCI`, `MACD`, `PSAR`).
-   **Hidden Layers:** A user-definable hidden layer structure (e.g., 1 layer with 12 neurons).
-   **Outputs (3 neurons):** Uses **one-hot encoding** for the three possible classes:
    -   `[1, 0, 0]` represents a **SELL** signal.
    -   `[0, 1, 0]` represents a **HOLD** signal.
    -   `[0, 0, 1]` represents a **BUY** signal.

#### Interactive Inference

After training with the `trade` command, you can test the network with the interactive tool. The output is a vector of probabilities (sigmoid), and the highest value indicates the network's final decision.
Interpretation of these vectors such as the highest value in the vecor such as {SELL,HOLD,BUY} helps make a decision on a trade.
**Example Interactive Test Session:**

```
> i trade

--- Interactive Inference Test ---
Network 'OCL_TRADE_Gate.nnw' loaded successfully.
--> Required Inputs: 8, Outputs: 3
Type 'q' or 'quit' to exit.

Enter 8 numbers separated by spaces > 1.1205 1.1285 1.1200 1.1290 0.0090 110.5 0.0018 1.1195
  Raw Output Vector: [0.0012, 0.0345, 0.9881]
  Interpreted Result: BUY

Enter 8 numbers separated by spaces > 1.1350 1.1270 1.1265 1.1355 0.0085 -115.3 -0.0016 1.1360
  Raw Output Vector: [0.9902, 0.0211, 0.0005]
  Interpreted Result: SELL
```
*The `Interpreted Result` shows the class corresponding to the output neuron with the highest activation, providing a clear and decisive prediction.*

## 💡 Code Structure

- **`main.cpp`:** Contains the entry point and creates the `NeuralNetworkTester` application object.
- **`NeuralNetworkTester.h/.cpp`:** The main application class that manages the user interface, command parsing, and orchestrates the training and testing processes.
- **`TrainingData` struct:** A simple struct to hold input/target pairs for training.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
