// DllExport.h (a new header file for convenience)
#pragma once

#ifdef GNEURAL_API_EXPORTS // This should be defined in your DLL project's preprocessor settings
#define GNEURAL_API __declspec(dllexport)
#else
#define GNEURAL_API __declspec(dllimport)
#endif