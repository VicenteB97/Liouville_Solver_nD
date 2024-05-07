import subprocess, json, os, shutil
from typing import Any

# 0.- Read the .json file where we defined the case:
def read_json() -> dict[str, Any] | None:
    with open('case_definition.json', 'r') as file:
        # Load the JSON data
        return json.load(file)

# 1.- Create the examples to be launched:
def write_header(sim_name: str, sim_data: dict[str, Any]):
    text_to_header = f'''
    #pragma once
    #define CASE "{sim_name}"

    // Choosing whether showing full or simplified timing information
    #define OUTPUT_INFO 0

    #define floatType {sim_data["simulation_parameters"]["floatType"]}

    // AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
    #define TOLERANCE_AMR       {sim_data["simulation_parameters"]["TOLERANCE_AMR"]}
    #define TOLERANCE_ConjGrad  {sim_data["simulation_parameters"]["TOLERANCE_ConjGrad"]}
    #define DISC_RADIUS         {sim_data["simulation_parameters"]["DISC_RADIUS"]}

    // State variables information
    #define PHASE_SPACE_DIMENSIONS  {sim_data["simulation_parameters"]["phase_space_dimensions"]}
    #define DOMAIN_INF {sim_data["simulation_parameters"]["DOMAIN_INF"]}
    #define DOMAIN_SUP {sim_data["simulation_parameters"]["DOMAIN_SUP"]}

    // Vector field definition
    // explanation: 
    #define VF_1    {sim_data["vector_field"]["VF_1"]}
    #define D_1     {sim_data["vector_field"]["D_1"]}
    #define VF_2    {sim_data["vector_field"]["VF_2"]}
    #define D_2     {sim_data["vector_field"]["D_2"]}

    #define VEC_FIELD {sim_data["vector_field"]["VEC_FIELD"]}
    #define DIVERGENCE {sim_data["vector_field"]["DIVERGENCE"]}

    static const char   IC_NAMES[PHASE_SPACE_DIMENSIONS] = {sim_data["initial_condition"]["IC_NAMES"]};
    static const bool   IC_isTRUNC[PHASE_SPACE_DIMENSIONS] = {sim_data["initial_condition"]["IC_isTRUNC"]};
    static const floatType   IC_InfTVAL[PHASE_SPACE_DIMENSIONS] = {sim_data["initial_condition"]["IC_InfTVAL"]};
    static const floatType   IC_SupTVAL[PHASE_SPACE_DIMENSIONS] = {sim_data["initial_condition"]["IC_SupTVAL"]};
    static const floatType	IC_MEAN[PHASE_SPACE_DIMENSIONS] = {sim_data["initial_condition"]["IC_MEAN"]};
    static const floatType	IC_STD[PHASE_SPACE_DIMENSIONS] = {sim_data["initial_condition"]["IC_STD"]};

    // Parameter information
    #define PARAM_SPACE_DIMENSIONS {sim_data["simulation_parameters"]["param_space_dimensions"]}
    static const char   _DIST_NAMES[PARAM_SPACE_DIMENSIONS] = {sim_data["parameters"]["DIST_NAMES"]};
    static const bool   _DIST_isTRUNC[PARAM_SPACE_DIMENSIONS] = {sim_data["parameters"]["DIST_isTRUNC"]};
    static const floatType  _DIST_InfTVAL[PARAM_SPACE_DIMENSIONS] = {sim_data["parameters"]["DIST_InfTVAL"]};
    static const floatType  _DIST_SupTVAL[PARAM_SPACE_DIMENSIONS] = {sim_data["parameters"]["DIST_SupTVAL"]};
    static floatType 		_DIST_MEAN[PARAM_SPACE_DIMENSIONS] = {sim_data["parameters"]["DIST_MEAN"]};
    static floatType 		_DIST_STD[PARAM_SPACE_DIMENSIONS] = {sim_data["parameters"]["DIST_STD"]};

    #define IMPULSE_TYPE 0
    #define INCLUDE_XTRA_PARAMS false'''

    with open('./src/include/Case_definition.cuh', 'w') as file:
        # Write the C++/CUDA header content into the file
        file.write(text_to_header)

def build_compile_execute(config: str, cores: str):
    my_path = os.getcwd() + f"\\build\\app\\{config}"

    if os.path.exists(f"{my_path}"):
        shutil.rmtree(f"{my_path}")
        print(f"{my_path} has been deleted")

    commands = ["cmake -S ./ -B ./build", f"cmake --build ./build --config {config} --parallel {cores}"]
    for command in commands:
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command '{command}': {e}")


if __name__ == "__main__":
    sim_cases = read_json()

    if sim_cases is not None:
        for case_name, case_props in sim_cases.items():
            write_header(case_name, case_props)
            # Compile and execute each case
            build_compile_execute("Release", "12")
