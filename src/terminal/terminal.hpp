#ifndef __TERMINAL_HPP__
#define __TERMINAL_HPP__

#include <cstdint>
#include <iostream>
#include <vector>
#include <string>

class terminal {
public:
	terminal() {
		__TERMINAL_WINDOW_SIZE = 90;
		__TERMINAL_CORNER_CHAR = "+";
		__FULL_SEPARATION_CHAR = "=";
		__SIMPLE_SEPARATION_CHAR = "-";
		__BORDER_CHAR = "|";
		__WRITING_COLOR = "W";
		__SEPARATION_FROM_BORDER = 2;
	};
	~terminal() {};
private:
	uint16_t __TERMINAL_WINDOW_SIZE;
	char __TERMINAL_CORNER_CHAR;
	char __FULL_SEPARATION_CHAR;
	char __SIMPLE_SEPARATION_CHAR;
	char __BORDER_CHAR;
	char __WRITING_COLOR;
	uint16_t __SEPARATION_FROM_BORDER;

public:
	void set_terminal_size(uint16_t new_size = 90) {
		__TERMINAL_WINDOW_SIZE = new_size;
	};
	void set_terminal_corner_char(char new_char = "+") {
		__TERMINAL_WINDOW_SIZE = new_char;
	};
	void set_terminal_full_sep_char(char new_char = "=") {
		__TERMINAL_WINDOW_SIZE = new_char;
	};
	void set_terminal_simple_sep_char(char new_char = "-") {
		__TERMINAL_WINDOW_SIZE = new_char;
	};
	void set_border_char(char new_char = "|") {
		__BORDER_CHAR = new_char;
	};
	void set_writing_color(char new_char = "W") {
		__TERMINAL_WINDOW_SIZE = new_char;
	};
	void set_separation_from_border(uint16_t new_size = 2) {
		__SEPARATION_FROM_BORDER = new_size;
	}

	void print_full_sep_line() {
		std::vector<char> print_buffer (__TERMINAL_WINDOW_SIZE, __FULL_SEPARATION_CHAR);
		print_buffer[0] = "+";
		print_buffer[__TERMINAL_WINDOW_SIZE - 1] = "+";

		std::cout << print_buffer << std::endl;
	};

	void print_simple_sep_line() {
		std::vector<char> print_buffer(__TERMINAL_WINDOW_SIZE, __SIMPLE_SEPARATION_CHAR);
		print_buffer[0] = "+";
		print_buffer[__TERMINAL_WINDOW_SIZE - 1] = "+";

		std::cout << print_buffer << std::endl;
	};

	void print_message(const std::string& print_message, char message_color = __WRITING_COLOR) {
		uint64_t message_length = print_message.length();
		uint16_t max_allowed_message_length = __TERMINAL_WINDOW_SIZE - 2 - (__SEPARATION_FROM_BORDER * 2);

		// Make the separation string
		std::string separation_str("");
		for (uint16_t k = 0; k < __SEPARATION_FROM_BORDER; k++) {
			separation_str += " ";
		}

		uint32_t line_breaks = ceil(message_length / max_allowed_message_length);

		std::vector<std::string> actual_print_messages(line_breaks, __BORDER_CHAR + separation_str);

		/*for (uint16_t k = 0; k < line_breaks; k++) {
			std::string partial_message = print_message
			actual_print_messages += 
		}*/
	}
};

#endif