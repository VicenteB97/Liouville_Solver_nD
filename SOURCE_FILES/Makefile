dependency:
	cd build && cmake .. --graphviz=graph.dot && dot -Tpng graph.dot -o graph_Image.png

prepare:
	rm -rf build
	mkdir build

bld:
	cd build && cmake ..

compile:
	cd build && cmake --build .

build_N_compile:
	cd build && cmake .. && cmake --build .

total:
	rm -rf build
	mkdir build
	cd build && cmake .. && cmake --build . && clear && ./Simulation
