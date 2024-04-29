compile_debug:
	rm -rf ./build
	mkdir ./build
	rm -rf ./output
	mkdir ./output
	cmake -S . -B ./build/Debug -DCMAKE_BUILD_TYPE=Debug
	cmake --build ./build/Debug

compile_release:
	rm -rf ./build
	mkdir ./build
	rm -rf ./output
	mkdir ./output
	cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release
	cmake --build ./build --parallel 12

total_debug:
	make compile_debug && cd ./build/Debug && clear && ./Simulation

total_release:
	make compile_release && cd ./build/Release && clear && ./Simulation

update_debug:
	cmake --build ./build && clear && cd ./build/Debug && ./Simulation
	
update_release:
	cmake --build ./build --parallel 12 && clear && cd ./build/Release && ./Simulation

cleanup:
	rm -rf ./build
	rm -rf ./output/*
	rm ./Visualization_matlab/*.asv