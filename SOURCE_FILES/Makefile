compile:
	rm -rf build
	mkdir build
	rm -rf ../SIMULATION_OUTPUT
	mkdir ../SIMULATION_OUTPUT
	cd build && cmake .. && cmake --build .

total:
	make compile && clear && ./Simulation

update:
	rm -rf ../SIMULATION_OUTPUT
	mkdir ../SIMULATION_OUTPUT
	cd build && cmake .. && cmake --build . && clear && ./Simulation

