compile_debug:
	rm -rf ../BUILD_FILES
	mkdir ../BUILD_FILES
	rm -rf ../SIMULATION_OUTPUT
	mkdir ../SIMULATION_OUTPUT
	cmake -S . -B ../BUILD_FILES/Debug -DCMAKE_BUILD_TYPE=Debug
	cmake --build ../BUILD_FILES/Debug

compile_release:
	rm -rf ../BUILD_FILES
	mkdir ../BUILD_FILES
	rm -rf ../SIMULATION_OUTPUT
	mkdir ../SIMULATION_OUTPUT
	cmake -S . -B ../BUILD_FILES/Release -DCMAKE_BUILD_TYPE=Release
	cmake --build ../BUILD_FILES/Release

Total_debug:
	make compile_debug && cd ../BUILD_FILES/Debug && clear && ./Simulation

Total_release:
	make compile_release && cd ../BUILD_FILES/Release && clear && ./Simulation

update_debug:
	cmake --build ../BUILD_FILES/Debug && clear && cd ../BUILD_FILES/Debug && ./Simulation
	
update_release:
	cmake --build ../BUILD_FILES/Release && clear && cd ../BUILD_FILES/Release && ./Simulation

cleanup:
	rm -rf ../BUILD_FILES
	rm -rf ../SIMULATION_OUTPUT/*
	rm ../Visualization_matlab/*.asv