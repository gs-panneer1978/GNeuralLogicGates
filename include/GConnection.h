#pragma once

class GConnection 
{

public:
	double getWeight(void) const;
	double getDeltaWeight(void);
	

	void setConnectionWeight(double weight);
	
private:
	double weight;
	double deltaWeight;

};

