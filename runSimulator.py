import subprocess, json, os, shutil, argparse
from typing import Any

# 0.- Read the .json file where we defined the case:
def read_json(filename: str) -> dict[str, Any] | None:
    with open(filename, 'r') as file:
        # Load the JSON data
        return json.load(file)

# 1.- Create the examples to be launched:
def write_header(sim_name: str, sim_data: dict[str, Any]):

    system_dimensions = int(sim_data["simulation_parameters"]["phase_space_dimensions"])
    write_components_str = []
    vec_field_str = 'vector_field'
    for i in range(system_dimensions):
        vector_field_id = f'VF_{i+1}'
        divergence_id = f'D_{i+1}'

        write_components_str.append(
            f'''
    #define {vector_field_id} (floatType){sim_data[vec_field_str][vector_field_id]}
    #define {divergence_id} {sim_data[vec_field_str][divergence_id]}
    '''
        )

    text_to_header = f'''
    #pragma once
    #define CASE "{sim_name}"

    // Choosing whether showing full or simplified timing information
    #define OUTPUT_INFO 0
    #define TERMINAL_INPUT_ALLOWED 0
    #define SAVING_TYPE "{sim_data["saving"]["type"]}"
    
    #define FIRST_FRAME {sim_data["saving"]["first_frame"]}
    #define LAST_FRAME {sim_data["saving"]["last_frame"]}

    #define floatType {sim_data["simulation_parameters"]["floatType"]}

    // AMR tolerance, Conjugate Gradient tolerance and number of discretization size for the radius of the RBFs
    #define TOLERANCE_AMR       {sim_data["simulation_parameters"]["TOLERANCE_AMR"]}
    #define TOLERANCE_ConjGrad  {sim_data["simulation_parameters"]["TOLERANCE_ConjGrad"]}
    #define DISC_RADIUS         {sim_data["simulation_parameters"]["DISC_RADIUS"]}

    // Phase space information
    #define PHASE_SPACE_DIMENSIONS  {sim_data["simulation_parameters"]["phase_space_dimensions"]}
    #define DOMAIN_INF {sim_data["simulation_parameters"]["DOMAIN_INF"]}
    #define DOMAIN_SUP {sim_data["simulation_parameters"]["DOMAIN_SUP"]}
    #define FINEST_DISCR_LVL {sim_data["simulation_parameters"]["disc_finest_level"]}
    
    // Timing definitions:
    #define INIT_TIME {sim_data["simulation_parameters"]["t0"]}
    #define FINAL_TIME {sim_data["simulation_parameters"]["tF"]}
    #define TIME_STEP {sim_data["simulation_parameters"]["delta_t"]}
    #define REINIT_STEPS {sim_data["simulation_parameters"]["reinit_steps"]}
    #define SAVING_STEPS {sim_data["simulation_parameters"]["saving_steps"]}

    // Use ad-hoc integrator? (ONLY FOR MATHIEU FOR NOW)
    #define SPECIAL_INTEGRATOR {sim_data.get("special_integrator", "false")}

    // Vector field definition
    // explanation:''' 

    for k in range(len(write_components_str)):
        text_to_header += write_components_str[k]

    text_to_header += f'''
    #define VEC_FIELD {sim_data["vector_field"]["VEC_FIELD"]}
    #define DIVERGENCE (floatType){sim_data["vector_field"]["DIVERGENCE"]}

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
    static floatType 		_DIST_N_SAMPLES[PARAM_SPACE_DIMENSIONS] = {sim_data["parameters"]["DIST_N_SAMPLES"]};'''

    if "impulse_parameters" not in sim_data.keys():
        text_to_header += '''

    #define IMPULSE_TYPE 0
    #define INCLUDE_XTRA_PARAMS false'''
    elif sim_data["impulse_parameters"]["type"] == 1:
        text_to_header += f'''

    #define IMPULSE_TYPE 1
    #define DiracDelta_impulseCount 3
    //	time | Imp | mean_vec  |   st. dev. | 	samples
    static double 		deltaImpulse_distribution_TIME[DiracDelta_impulseCount] = {sim_data["impulse_parameters"]["IMPULSE_TIMES"]};
    static const char   deltaImpulse_distribution_NAMES[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = {sim_data["impulse_parameters"]["DIST_NAMES"]};
    static const bool   deltaImpulse_distribution_isTRUNC[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = {sim_data["impulse_parameters"]["DIST_isTRUNC"]};
    static const floatType  deltaImpulse_distribution_InfTVAL[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = {sim_data["impulse_parameters"]["DIST_InfTVAL"]};
    static const floatType  deltaImpulse_distribution_SupTVAL[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = {sim_data["impulse_parameters"]["DIST_SupTVAL"]};
    static floatType 		deltaImpulse_distribution_MEAN[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = {sim_data["impulse_parameters"]["DIST_MEAN"]};
    static floatType 		deltaImpulse_distribution_STD[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = {sim_data["impulse_parameters"]["DIST_STD"]};
    static const int 	deltaImpulse_distribution_SAMPLES[DiracDelta_impulseCount * PHASE_SPACE_DIMENSIONS] = {sim_data["impulse_parameters"]["DIST_N_SAMPLES"]};

    #define INCLUDE_XTRA_PARAMS false
    '''

    with open('./src/include/Case_definition.cuh', 'w') as file:
        # Write the C++/CUDA header content into the file
        file.write(text_to_header)

# 2.- Build, compile and execute tests
def build_compile_execute(config: str = "Release", cores: str = "12", clean_start: bool = True):
    my_path = os.getcwd() + f"/build/app/{config}"
    
    commands = ["cmake -S ./ -B ./build"]

    if os.path.exists(f"{my_path}") and clean_start:
        shutil.rmtree(f"{my_path}")
        print(f"{my_path} has been deleted")
    else:
        commands.append(f"cmake --build ./build --target clean --config {config}")
        
    commands.extend([
            f"cmake --build ./build --config {config} --parallel {cores}",
            "cls"
        ])
    
    for command in commands:
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command '{command}': {e}")
            
    try:        
        subprocess.run(f"{my_path}/Simulation.exe", shell=True, check=True)
    except subprocess.CalledProcessError as e:
            print(f"Error executing the simulation. Code: {e}")

# Note that this does not mean you can't define the case_definition.cuh by yourself!
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Give simulation case file name.')
    parser.add_argument('filename', type=str, help='the filename to read')
    args = parser.parse_args()

    filename = args.filename
    
    sim_cases = read_json("./Definition_examples/" + filename)

    if sim_cases is not None:
        for case_name, case_props in sim_cases.items():
            write_header(case_name, case_props)
            # Compile and execute each case
            build_compile_execute()

