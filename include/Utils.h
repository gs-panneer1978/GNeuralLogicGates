#pragma once

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers

#include <windows.h>
#include <vector>
#include <string>
#include <fstream>
#include <streambuf>
#include <stdexcept>






//---------------------------------------------------------//
//--- Framework Utils ---//
//---------------------------------------------------------//
// Converts a std::wstring to a UTF-8 encoded std::string
std::string WStringToUTF8(const std::wstring& wstr);
/**
 * @brief Writes the binary representation of a double to a file.
 * @param hFile A valid file handle from CreateFile with GENERIC_WRITE access.
 * @param value The double value to write.
 * @return TRUE (1) on success, FALSE (0) on failure.
 */
bool FileWriteDouble(HANDLE hFile, double value);
/**
 * @brief Reads the binary representation of a double from a file.
 * @param hFile A valid file handle from CreateFile with GENERIC_READ access.
 * @param pValue A pointer to a double variable that will receive the value.
 * @return TRUE (1) on success, FALSE (0) on failure (e.g., end of file).
 */
bool FileReadDouble(HANDLE hFile, double* pValue);

//USING STREAMS IN C++ FOR FILE OPERATIONS

/**
 * @brief Writes the binary representation of a double to a file.
 *
 */
bool StreamWriteDouble(std::ofstream& outFile, double value);
/**
 * @brief Reads the binary representation of a double from a C++ input file stream.
 * @param inFile A reference to a valid std::ifstream object, opened in binary mode.
 * @param value A reference to a double variable that will receive the value.
 * @return true on success, false on failure (e.g., end of file).
 *
 */
bool StreamReadDouble(std::ifstream& inFile, double& value);

