#pragma once
#include <stdlib.h>
#include <vector>
#include <ctime>
//#include "GNeuron.h"

using namespace std;

typedef enum 
{
	TANH = 0,
	SIGMOID = 1,
	RELU = 2
} ENUM_ACTIVATION;

typedef vector<unsigned> VectorUnsigned;
typedef vector<int> VectorInt;
typedef vector<double> VectorDouble;
typedef vector<float> VectorFloat;


typedef vector<size_t> Topology;
//typedef vector<GNeuron> Memory;


typedef enum 
{
	SGD,
	ADAM
} ENUM_OPTIMIZATION;
 

typedef enum 
{
	WEIGHTS,
	DELTA_WEIGHTS,
	OUTPUT,
	GRADIENT,
	FIRST_MOMENTUM,
	SECOND_MOMENTUM
} ENUM_BUFFERS;

class DateTime {
public: 
	//tm* getGMTTime(void) { return gmtime_s(&now); }
	//char* getAscTime(void) { return asctime_s(&now); }
	char* getCurrentTime(void) { return dt; }
private:
	time_t now = time(0);
	char* dt = ctime(&now);

};


class NNetTrainingData {
public:

private:
	VectorDouble m_input;
	VectorDouble m_output;


};
#ifdef MAKEDLL
#  define GN_EXPORT __declspec(dllexport)
#else
#  define GN_EXPORT __declspec(dllimport)
#endif








