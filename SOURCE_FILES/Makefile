compile:
	rm -rf ../BUILD
	mkdir ../BUILD
	rm -rf ../SIMULATION_OUTPUT
	mkdir ../SIMULATION_OUTPUT
	cmake -B ../BUILD
	cmake --build ../BUILD

total:
	make compile && cd ../BUILD && clear && ./Simulation

update:
	rm -rf ../SIMULATION_OUTPUT
	mkdir ../SIMULATION_OUTPUT
	cmake -B ../BUILD
	cmake --build ../BUILD && clear && cd ../BUILD && ./Simulation

cleanup:
	rm -rf ../BUILD
	rm ../SIMULATION_OUTPUT/*
	rm -i ../Visualization_matlab/*.asv

