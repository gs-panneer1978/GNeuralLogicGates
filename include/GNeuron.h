#pragma once
#include <stdlib.h>
#include <vector>
#include "GTypes.h"
#include "GObject.h"
#include "GConnection.h"
#include "GNeuralConnection.h"
#include "constants.h"


using namespace std;
class GNeuron;

//std::vector<GNeuron*> Layer;
typedef vector<GNeuron> Layer;
// Use unique_ptr to manage memory automatically.
//typedef std::vector<std::unique_ptr<GNeuron>> Layer;

//template<class T>
class GNeuron : public GObject
{
public:
					GNeuron() : m_outputVal(0.0), m_myIndex(0), m_gradient(0.0), m_activationType(SIGMOID) {}
	// Constructor for creating a neuron with a specified number of outputs and its index in the layer
					GNeuron(unsigned numOutputs, unsigned myIndex, ENUM_ACTIVATION activationType);

	int				GetTypeID() const override { return defNeuron; }
	void			setOutputVal(double val) { m_outputVal = val; }
	double			getOutputVal(void) const { return m_outputVal; }
	void			feedForward(Layer& prevLayer);
	void			calcOutputGradients(double targetVal);
	void			calcHiddenGradients(const Layer& nextLayer);
	void			updateInputWeights(Layer& prevLayer);

	void			SetActivationType(ENUM_ACTIVATION activation_type) { m_activationType = activation_type; }
	
	const			std::vector<GNeuralConnection>& getOutputWeights() const { return m_outputWeights; }
	double			getGradient(void) const { return m_gradient; }

	static void		setEta(double val) { eta = val; }
	static void		setAlpha(double val) { alpha = val; }

	GNeuralConnection&						getConnection(size_t index);
	const GNeuralConnection&				getConnection(size_t index) const;
	
private:
	static double	eta;
	static double	alpha;
	double			transferFunction(double x);
	double			transferFunctionDerivative(double x);
	double			randomWeight(void) { return rand() / double(RAND_MAX); }
	double			sumDOW(const Layer& nextLayer) const;
	double			m_outputVal;
	
	unsigned		m_myIndex;
	double			m_gradient;

	vector<GNeuralConnection> m_outputWeights;
	ENUM_ACTIVATION m_activationType; // Default activation type
};

