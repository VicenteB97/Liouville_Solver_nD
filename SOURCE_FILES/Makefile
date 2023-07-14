compile:
	rm -rf ../BUILD_FILES
	mkdir ../BUILD_FILES
	rm -rf ../SIMULATION_OUTPUT
	mkdir ../SIMULATION_OUTPUT
	cmake -S . -B ../BUILD_FILES
	cmake --build ../BUILD_FILES

Total:
	make compile && cd ../BUILD_FILES && clear && ./Simulation

update:
	cmake --build ../BUILD_FILES && clear && cd ../BUILD_FILES && ./Simulation

cleanup:
	rm -rf ../BUILD_FILES
	rm -rf ../SIMULATION_OUTPUT/*
	rm ../Visualization_matlab/*.asv