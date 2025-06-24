#pragma once
#include "constants.h"
#include "GObject.h"
//#include <unistd.h> // for write and read functions

class GNeuralConnection : public GObject
{

private:
	double			weight;
	double			deltaWeight;
	double			mt;
	double			vt;

public:
					GNeuralConnection() {}
					GNeuralConnection(double w) { weight = w; deltaWeight = 0; mt = 0; vt = 0; }
					~GNeuralConnection() {};
					
					int GetTypeID() const override { return defConnect; } // Override to return the type ID for this connection
	//--- methods for working with files

	virtual bool    Save(HANDLE file_handle) const;
	virtual bool    Load(HANDLE file_handle);
	virtual bool    Save(std::ofstream& outFile) const;
	virtual bool    Load(std::ifstream& inFile);
	virtual int     Type(void)   const { return defConnect; }

	double			getWeight(void) const;
	double			getDeltaWeight(void);

	void			setConnectionWeight(double weight);
	void			setDeltaWeight(double delta) { deltaWeight = delta; }	



};

