#ifndef __TERMINAL_HPP__
#define __TERMINAL_HPP__

#include "include/headers.hpp"
#include "include/indicators/progress_bar.hpp"

class terminal {
private:
	uint16_t __TERMINAL_WINDOW_SIZE;
	char __TERMINAL_CORNER_CHAR;
	char __FULL_SEPARATION_CHAR;
	char __SIMPLE_SEPARATION_CHAR;
	char __BORDER_CHAR;
	char __WRITING_COLOR;
	uint16_t __SEPARATION_FROM_BORDER;
	// indicators::ProgressBar m_statusBar;

public:
	// Constructor
	terminal() : __TERMINAL_WINDOW_SIZE(90),
		__TERMINAL_CORNER_CHAR('+'),
		__FULL_SEPARATION_CHAR('='),
		__SIMPLE_SEPARATION_CHAR('-'),
		__BORDER_CHAR('|'),
		__WRITING_COLOR('W'),
		__SEPARATION_FROM_BORDER(2){
		// The commented section of the progress bar initialization
		// can be re-added if indicators library is used.
		//m_statusBar{
		//	indicators::option::BarWidth{35},
		//	indicators::option::ForegroundColor{indicators::Color::yellow},
		//	indicators::option::ShowElapsedTime{true},
		//	indicators::option::ShowRemainingTime{false},
		//	indicators::option::PrefixText{"[INFO] Running..."},
		//	indicators::option::Start{"["},
		//	indicators::option::Fill{"*"},
		//	indicators::option::Lead{"*"},
		//	indicators::option::Remainder{"-"},
		//	indicators::option::End{"]"}
		//};
	};

	// Destructor
	~terminal() {};

	// Setters
	void set_terminal_size(uint16_t new_size = 90) {
		__TERMINAL_WINDOW_SIZE = new_size;
	};

	void set_terminal_corner_char(const char* new_char = "+") {
		__TERMINAL_CORNER_CHAR = new_char[0];  // Use the first character of the string
	};

	void set_terminal_full_sep_char(const char* new_char = "=") {
		__FULL_SEPARATION_CHAR = new_char[0];  // Fix incorrect assignment
	};

	void set_terminal_simple_sep_char(const char* new_char = "-") {
		__SIMPLE_SEPARATION_CHAR = new_char[0];  // Fix incorrect assignment
	};

	void set_border_char(const char* new_char = "|") {
		__BORDER_CHAR = new_char[0];  // Fix incorrect assignment
	};

	void set_writing_color(const char* new_char = "W") {
		__WRITING_COLOR = new_char[0];  // Fix incorrect assignment
	};

	void set_separation_from_border(uint16_t new_size = 2) {
		__SEPARATION_FROM_BORDER = new_size;
	}

	// Methods to print separator lines
	void print_full_sep_line() {
		std::string print_buffer(__TERMINAL_WINDOW_SIZE, __FULL_SEPARATION_CHAR);
		print_buffer[0] = __TERMINAL_CORNER_CHAR;
		print_buffer[__TERMINAL_WINDOW_SIZE - 1] = __TERMINAL_CORNER_CHAR;

		std::cout << print_buffer << std::endl;
	};

	void print_simple_sep_line() {
		std::string print_buffer(__TERMINAL_WINDOW_SIZE, __SIMPLE_SEPARATION_CHAR);
		print_buffer[0] = __TERMINAL_CORNER_CHAR;
		print_buffer[__TERMINAL_WINDOW_SIZE - 1] = __TERMINAL_CORNER_CHAR;

		std::cout << print_buffer << std::endl;
	};

	// Print message with padding
	void print_message(const std::string& print_message, const std::string& endlineStr = "\n", char message_color = 'W') {
		uint64_t message_length = print_message.length();
		uint16_t border_size = 2;  // One on each side

		// Calculate maximum allowed message length inside the terminal
		uint16_t max_allowed_message_length = __TERMINAL_WINDOW_SIZE - border_size - (__SEPARATION_FROM_BORDER * 2);

		// Split message into lines if necessary
		std::string separation_str(__SEPARATION_FROM_BORDER, ' ');  // Padding spaces

		// Print message, wrap it if needed
		if (message_length <= max_allowed_message_length) {
			std::cout << __BORDER_CHAR << separation_str << print_message << std::string(max_allowed_message_length - message_length, ' ') << separation_str << __BORDER_CHAR << endlineStr;
		}
		else {
			// Handle line breaks
			size_t current_pos = 0;
			while (current_pos < message_length) {
				std::string message_part = print_message.substr(current_pos, max_allowed_message_length);
				std::cout << __BORDER_CHAR << separation_str << message_part;
				if (message_part.length() < max_allowed_message_length) {
					std::cout << std::string(max_allowed_message_length - message_part.length(), ' ');
				}
				std::cout << separation_str << __BORDER_CHAR << std::endl;
				current_pos += max_allowed_message_length;
			}
		}
		std::cout.flush();
	};

	// Simulation status methods
	void update_simulation_status(const uint32_t currentIteration, const uint32_t totalIterations) {
		std::string updateMessage = "[RUNNING] Current progress: (" + std::to_string(currentIteration + 1) + "/" + std::to_string(totalIterations) + ")";
		this->print_message(updateMessage, "\r");
		// Update progress bar if applicable (commented section for progress bar usage)
	};

	void simulation_completed() {
		this->print_message("");
		this->print_message("Simulation completed successfully.");
	};
};

void printEntryMessage() {
	terminal myTerminal;
	const std::string str_project_version{ project_version };

	myTerminal.print_full_sep_line();
	myTerminal.print_message("Welcome to the Liouville Eq. Simulator. You are using version " + str_project_version);
	myTerminal.print_simple_sep_line();
	myTerminal.print_message("Starting simulation using GPU " + gpu_device.deviceProperties.name);
	myTerminal.print_message("Properties:");
	myTerminal.print_message("  - Global memory (GB): " + std::to_string(gpu_device.deviceProperties.totalGlobalMem / 1024 / 1024 / 1024));
	myTerminal.print_message("  - Max. memory bus width (b): " + std::to_string(gpu_device.deviceProperties.memoryBusWidth));
	myTerminal.print_message("  - Async. engine type (0/1/2): " + std::to_string(gpu_device.deviceProperties.asyncEngineCount));
	myTerminal.print_full_sep_line();
}

#endif