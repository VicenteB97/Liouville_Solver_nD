# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/Vicentin/source/repos/Liouville_Solver_nD/SOURCE_FILES

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/Vicentin/source/repos/Liouville_Solver_nD/BUILD_FILES/Release

# Include any dependencies generated for this target.
include CMakeFiles/Solver_Lib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Solver_Lib.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Solver_Lib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Solver_Lib.dir/flags.make

CMakeFiles/Solver_Lib.dir/FULL_SIMULATION.cu.o: CMakeFiles/Solver_Lib.dir/flags.make
CMakeFiles/Solver_Lib.dir/FULL_SIMULATION.cu.o: /mnt/c/Users/Vicentin/source/repos/Liouville_Solver_nD/SOURCE_FILES/FULL_SIMULATION.cu
CMakeFiles/Solver_Lib.dir/FULL_SIMULATION.cu.o: CMakeFiles/Solver_Lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/Vicentin/source/repos/Liouville_Solver_nD/BUILD_FILES/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/Solver_Lib.dir/FULL_SIMULATION.cu.o"
	/usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/Solver_Lib.dir/FULL_SIMULATION.cu.o -MF CMakeFiles/Solver_Lib.dir/FULL_SIMULATION.cu.o.d -x cu -c /mnt/c/Users/Vicentin/source/repos/Liouville_Solver_nD/SOURCE_FILES/FULL_SIMULATION.cu -o CMakeFiles/Solver_Lib.dir/FULL_SIMULATION.cu.o

CMakeFiles/Solver_Lib.dir/FULL_SIMULATION.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Solver_Lib.dir/FULL_SIMULATION.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Solver_Lib.dir/FULL_SIMULATION.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Solver_Lib.dir/FULL_SIMULATION.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target Solver_Lib
Solver_Lib_OBJECTS = \
"CMakeFiles/Solver_Lib.dir/FULL_SIMULATION.cu.o"

# External object files for target Solver_Lib
Solver_Lib_EXTERNAL_OBJECTS =

libSolver_Lib.a: CMakeFiles/Solver_Lib.dir/FULL_SIMULATION.cu.o
libSolver_Lib.a: CMakeFiles/Solver_Lib.dir/build.make
libSolver_Lib.a: CMakeFiles/Solver_Lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/Vicentin/source/repos/Liouville_Solver_nD/BUILD_FILES/Release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA static library libSolver_Lib.a"
	$(CMAKE_COMMAND) -P CMakeFiles/Solver_Lib.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Solver_Lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Solver_Lib.dir/build: libSolver_Lib.a
.PHONY : CMakeFiles/Solver_Lib.dir/build

CMakeFiles/Solver_Lib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Solver_Lib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Solver_Lib.dir/clean

CMakeFiles/Solver_Lib.dir/depend:
	cd /mnt/c/Users/Vicentin/source/repos/Liouville_Solver_nD/BUILD_FILES/Release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/Vicentin/source/repos/Liouville_Solver_nD/SOURCE_FILES /mnt/c/Users/Vicentin/source/repos/Liouville_Solver_nD/SOURCE_FILES /mnt/c/Users/Vicentin/source/repos/Liouville_Solver_nD/BUILD_FILES/Release /mnt/c/Users/Vicentin/source/repos/Liouville_Solver_nD/BUILD_FILES/Release /mnt/c/Users/Vicentin/source/repos/Liouville_Solver_nD/BUILD_FILES/Release/CMakeFiles/Solver_Lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Solver_Lib.dir/depend

