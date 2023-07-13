compile:
	rm -rf ../BUILD_FILES
	mkdir ../BUILD_FILES
	rm -rf ../SIMULATION_OUTPUT
	mkdir ../SIMULATION_OUTPUT
	cmake -DCMAKE_TOOLCHAIN_FILE="target_toolchain.cmake" -S . -B ../BUILD_FILES
	cmake --build ../BUILD_FILES

total:
	make compile && cd ../BUILD_FILES && clear && ./Simulation

update:
	rm -rf ../SIMULATION_OUTPUT
	mkdir ../SIMULATION_OUTPUT
	cmake -DCMAKE_TOOLCHAIN_FILE="target_toolchain.cmake" -S . -B ../BUILD_FILES
	cmake --build ../BUILD_FILES && clear && cd ../BUILD_FILES && ./Simulation

cleanup:
	rm -rf ../BUILD_FILES
	rm ../SIMULATION_OUTPUT/*
	rm -i ../Visualization_matlab/*.asv

