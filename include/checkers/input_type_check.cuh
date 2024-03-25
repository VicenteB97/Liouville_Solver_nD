#pragma once

#include "../headers.cuh"

/// @brief 
/// @param inputTerminal 
/// @return 
inline bool isNumeric(const std::string& inputTerminal){

	// Iterate through every character in the string and check if they're numbers
	int16_t strLength = inputTerminal.size(), CharPosition = 0;
	bool isDecimal = false;

	// It's OK if the first term is a minus sign
	if(inputTerminal[0] == '-'){
		CharPosition++;
	}
	// It's OK if the first term is a dot character
	else if(inputTerminal[0] == '.'){
		CharPosition++;
		isDecimal = true;
	}

	// Go through all characters in the string
	while(CharPosition < strLength){

		// Check if the character is a digit
		if(!std::isdigit(inputTerminal[CharPosition])){
			
			// Check if it's the dot character and wether it has already appeared
			if(inputTerminal[CharPosition] == '.' && !isDecimal){
				isDecimal = true;
				CharPosition++;
			}
			else{
				return false;
			}
		}
		CharPosition++;
	}
	return true;

}

inline int16_t intCheck(bool& getAnswer, const std::string& inputTerminal, const std::string& errMessage = "Undefined error occured.\n", const INT non_accepted = 0, const INT minArg = std::numeric_limits<INT>::lowest(), 
						const INT maxArg = std::numeric_limits<INT>::max()) {
	
	if (!isNumeric(inputTerminal)) { std::cout << "Error: Non-numerical inputs not allowed. "; }
	else {

		INT temp = std::stoi(inputTerminal);

		if (temp == -1) {
			std::cout << "Definition error in file: " << __FILE__ << "\nLine: " << __LINE__ << "\nExiting simulation.\n";
			return -1;
		}

		if (temp < minArg || temp == non_accepted || temp > maxArg) {
			std::cout << errMessage;
		}
		else {
			getAnswer = false;
		}
	}

	return 0;
	
}

inline int16_t doubleCheck(bool& getAnswer, const std::string& inputTerminal, const std::string& errMessage = "Undefined error occured.\n", const double non_accepted = 0, const double minArg = std::numeric_limits<double>::lowest(),
							const double maxArg = std::numeric_limits<double>::max()) {

	if (!isNumeric(inputTerminal)) { std::cout << "Error: Non-numerical inputs not allowed. "; }
	else {

		double temp = std::stod(inputTerminal);

		if (temp == -1) {
			std::cout << "Definition error in file: " << __FILE__ << "\nLine: " << __LINE__ << "\nExiting simulation.\n";
			return -1;
		}

		if (temp == non_accepted || temp < minArg || temp > maxArg) {
			std::cout << errMessage;
		}
		else {
			getAnswer = false;
		}
	}

	return 0;
	
}