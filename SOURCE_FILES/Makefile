win32_compile:
	rmdir ..\BUILD_FILES
	mkdir ..\BUILD_FILES
	rmdir ..\SIMULATION_OUTPUT
	mkdir ..\SIMULATION_OUTPUT
	cmake -G "Visual Studio 17 2022" -A Win64 -S . -B ..\BUILD_FILES
	cmake --build ..\BUILD_FILES

Linx_compile:
	rm -rf ../BUILD_FILES
	mkdir ../BUILD_FILES
	rm -rf ../SIMULATION_OUTPUT
	mkdir ../SIMULATION_OUTPUT
	cmake -S . -B ../BUILD_FILES
	cmake --build ../BUILD_FILES

win32_total:
	make win32_compile && cd ..\BUILD_FILES && clear && ./Simulation

Linx_total:
	make Linx_compile && cd ../BUILD_FILES && clear && ./Simulation

update:
	cmake --build ../BUILD_FILES && clear && cd ../BUILD_FILES && ./Simulation

cleanup:
	rm -rf ../BUILD_FILES
	rm -rf ../SIMULATION_OUTPUT/*
	rm ../Visualization_matlab/*.asv

