#ifndef __SIM_DATA_CUH__
#define __SIM_DATA_CUH__

#include "Domain.cuh"
#include "Probability.cuh"

// Time + impulse: ----------------------------------------------
class Time_instants {
public:
    double 	time;
    bool 	impulse;

    bool operator < (const Time_instants& other) const {
        return (time < other.time);
    }
};

// Simulation log for further analysis
class Logger {
public:
    std::vector<double> subFrame_time;
    std::vector<int32_t> ConvergenceIterations;

    // default constructor
    Logger() {
        subFrame_time[0] = 0;
        ConvergenceIterations[0] = 0;
    }

    Logger(const uint32_t& size) {
        subFrame_time.resize(5 * size, (double)0);
        ConvergenceIterations.resize(size, (int32_t)0);
    }

    // Get the total frame time:
    inline double FrameTime(const uint32_t& timeStep) const {
        return (subFrame_time[5 *  timeStep] + subFrame_time[5 *  timeStep + 1] + subFrame_time[5 *  timeStep + 2] + subFrame_time[5 *  timeStep + 3] + subFrame_time[5 *  timeStep + 4]);
    }

    // Write to Command Line
    void writeToCLI(const uint16_t& Log_Lvl, const uint32_t& time_step) const {

        switch(Log_Lvl){
            case 1:
                // Just show the total timestep + memory information (@ the GPU)
                std::cout << "Total frame time: " << this->FrameTime(time_step) << "s.\n";
                break;

            case 2:
                // Show timing information
                std::cout << "AMR timing: " << subFrame_time[5 *  time_step] << " s.\n";
                std::cout << "Point search timing: " << subFrame_time[5 *  time_step + 2] << " s.\n";
                std::cout << "Interpolation timing: " << subFrame_time[5 *  time_step + 3] << " s. CG Iterations: " << ConvergenceIterations[time_step] << ".\n";
                std::cout << "Advection timing: " << subFrame_time[5 *  time_step + 1] << " s.\n";
                std::cout << "Reinitialization timing: " << subFrame_time[5 *  time_step + 4] << " s.\n";
                std::cout << "Total frame time: " << this->FrameTime(time_step) << " s.\n";
                break;

            default:
                break;
        }
    }

    void writeToFile() const {
        // Complete function to write into a log file!
        std::string Log_fileName = LOG_OUTPUT_relPATH;
        Log_fileName.append("LogFile.csv");	// default filename

        std::ofstream log_file(Log_fileName, std::ios::out);
        assert(log_file.is_open());

        std::string temp = "AMR, Advection, Point search, Interpolation, CG Iterations, Reinitialization, Total";

        log_file << temp << "\n";

        for (uint32_t i = 0; i < ConvergenceIterations.size(); i++) {

            temp = std::to_string(subFrame_time[5 * i + 0]);
            temp.append(",");
            temp.append(std::to_string(subFrame_time[5 * i + 1]));
            temp.append(",");
            temp.append(std::to_string(subFrame_time[5 * i + 2]));
            temp.append(",");
            temp.append(std::to_string(subFrame_time[5 * i + 3]));
            temp.append(",");
            temp.append(std::to_string(ConvergenceIterations[i]));
            temp.append(",");
            temp.append(std::to_string(subFrame_time[5 * i + 4]));
            temp.append(",");
            temp.append(std::to_string(this->FrameTime(i)));
                        
            log_file << temp << "\n";
        }

        log_file.close();
    }

    void writeToFile(const std::string& File_name) const {
        // Complete function to write into a log file!
        std::string Log_fileName = LOG_OUTPUT_relPATH;
        Log_fileName.append(File_name);

        std::ofstream log_file(Log_fileName, std::ios::out);
        assert(log_file.is_open());

        std::string temp = "AMR, Advection, Point search, Interpolation, CG Iterations, Reinitialization, Total";

        log_file << temp << "\n";

        for (uint32_t i = 0; i < ConvergenceIterations.size(); i++) {

            temp = std::to_string(subFrame_time[5 * i + 0]);
            temp.append(",");
            temp.append(std::to_string(subFrame_time[5 * i + 1]));
            temp.append(",");
            temp.append(std::to_string(subFrame_time[5 * i + 2]));
            temp.append(",");
            temp.append(std::to_string(subFrame_time[5 * i + 3]));
            temp.append(",");
            temp.append(std::to_string(ConvergenceIterations[i]));
            temp.append(",");
            temp.append(std::to_string(subFrame_time[5 * i + 4]));
            temp.append(",");
            temp.append(std::to_string(this->FrameTime(i)));

            log_file << temp << "\n";
        }

        log_file.close();
    }
};



// namespace IVP_LiouvilleSolver{

    
//     // Full simulation information
//     template<uint16_t PHASE_SPACE_DIM, uint16_t PARAM_SPACE_DIM, class T>
//     class Simulation_info{
//     public:
//         // Domain information
//         grid<PHASE_SPACE_DIM, T> Problem_Domain;

//         // Probabilistic information about ICs and parameters
//         std::vector<Distributions<T>> Param_distributions;
        
//         // Time instants where we will get PDF frames
//         std::vector<Time_instants> timeInstantsVec;

//         // Log array holding timing information
//         Logger Log;

//         // Array holding the PDF frames
//         std::vector<T> storeFrames;

//         // Time elapsed
//         double simulationDuration;

//         // Default constructor
//         Simulation_info<PHASE_SPACE_DIM, PARAM_SPACE_DIM, T>(){}

//         // Parametric constructor
//         Simulation_info<PHASE_SPACE_DIM, PARAM_SPACE_DIM, T> (uint32_t& size ){
//             timeInstantsVec.resize(size);
//             storeFrames.resize(size);
//         }

//     // Now we write the methods!
//     public:
//         void LogToFile(){
//             Log.writeToFile();
//         }

//         void LogToFile(const std::string& fileName){
//             Log.writeToFile(fileName);
//         }

        // int16_t WriteFramesToFile(){
        //     // We have added the capability of automatically detecting the number of 1 GB files where we can store the simulation output
        //     bool saving_active = true;		// see if saving is still active
        //     int16_t error_check = 0;

        //     const uint64_t MEM_2_STORE		= storeFrames.size() * sizeof(float);
            
        //     UINT number_of_frames_needed 	= MEM_2_STORE / Problem_Domain.Total_Nodes() / sizeof(float);
        //     uint64_t max_frames_file 		= (uint64_t)MAX_FILE_SIZE_B / Problem_Domain.Total_Nodes() / sizeof(float);
        //     UINT number_of_files_needed  	= floor((number_of_frames_needed - 1) / max_frames_file) + 1;
            
        //     char ans;
        //     std::cout << "Simulation time: " << simulationDuration << " seconds. ";
            
        //     if(number_of_files_needed == 0){
        //         std::cout << "There has been a problem. No memory written. Exiting simulation.\n";
        //         saving_active = false;
        //         error_check = -1;
        //     }

        //     while(saving_active){	
        //         std::cout << "Total memory of simulation: " << (float) MEM_2_STORE / 1024/1024 << " MB. \n";
        //         std::cout << number_of_files_needed << " files required for total storage. Total frames: " << number_of_frames_needed << ", with frames per file: " << max_frames_file << " \n";
        //         std::cout << "Write? (Y = Yes(total), N = No, P = Partial): ";
        //         std::cin >> ans;

        //         while((ans != 'N') && (ans != 'n') && (ans != 'Y') && (ans != 'y') && (ans != 'P') && (ans != 'p')){
        //             std::cout << "Incorrect option. Choose one of the following (NOT case sensitive: Y = Yes, N = No, P = Partial): ";
        //             std::cin >> ans;
        //         }


        //         if ((ans != 'N') && (ans != 'n')) {

        //             INT frames_init = 0, frames_end = number_of_files_needed - 1;
        //             bool condition = false;

        //             if((ans == 'P') || (ans == 'p')){
        //                 while(!condition){
        //                     std::cout << "Initial frame (must be >= 0): ";
        //                     std::cin >> frames_init;
        //                     std::cout << "Final frame (must be < "<< number_of_frames_needed <<"): ";
        //                     std::cin >> frames_end;

        //                     if(frames_init < 0 || frames_end >= number_of_frames_needed || frames_init > frames_end){

        //                         if(frames_init == -1 || frames_end == -1){
        //                             std::cout << "Exiting simulation without saving simulation results...\n";
        //                             return -1;
        //                         }
        //                         std::cout << "Check numbers, something's not right...\n";
        //                     }
        //                     else{
        //                         condition = true;
        //                         number_of_frames_needed = frames_end - frames_init + 1;
        //                         number_of_files_needed  = floor((number_of_frames_needed - 1) / max_frames_file) + 1;
        //                     }
        //                 }
        //             }
        //             else{
        //                 frames_init = 0, frames_end = number_of_files_needed - 1;
        //             }

        //             #pragma omp parallel for
        //             for(int16_t k = 0; k < number_of_files_needed; k++){

        //                 UINT frames_in_file = fmin(max_frames_file, number_of_frames_needed - k * max_frames_file);

        //                 std::string temp_str = std::to_string((UINT)k);

        //             // SIMULATION INFORMATION FILE
        //                 std::string relavtive_pth = SIM_OUTPUT_relPATH;
        //                 relavtive_pth.append("Simulation_info_");
        //                 relavtive_pth.append(temp_str);
        //                 relavtive_pth.append(".csv");

        //                 std::ofstream file1(relavtive_pth, std::ios::out);
        //                 assert(file1.is_open());

        //                 file1 << Problem_Domain.Total_Nodes() << "," << Problem_Domain.Nodes_per_Dim << ",";

        //                 for (UINT d = 0; d < PHASE_SPACE_DIMENSIONS; d++)
        //                           file1 << Problem_Domain.Boundary_inf.dim[d] << "," << Problem_Domain.Boundary_sup.dim[d] << ",";
        //                 }
        //                 for (uint16_t d = 0; d < PARAM_SPACE_DIMENSIONS; d++) {
        //                     file1 << Param_dist[d].num_Samples << ",";
        //                 }
        //                 file1 << duration.count() << "\n";

        //                 #if IMPULSE_TYPE == 0 || IMPULSE_TYPE ==1
        //                 for (UINT i = k * max_frames_file + frames_init; i < k * max_frames_file + frames_in_file + frames_init - 1; i++) {
        //                     file1 << time_vector[i].time << ",";
        //                 }
        //                 file1 << time_vector[k * max_frames_file + frames_in_file + frames_init - 1].time;

        //                 #elif IMPULSE_TYPE == 2
        //                 file1 << time_vector[k * max_frames_file + frames_init].time << ",";

        //                 for (UINT i = k * max_frames_file + 1 + frames_init; i < k * max_frames_file + frames_init + frames_in_file; i++) {
        //                     if (abs(time_vector[i].time - time_vector[i - 1].time) > pow(10, -7)) {
        //                         file1 << time_vector[i].time << ",";
        //                     }
        //                     else if (i == k * max_frames_file + frames_in_file + frames_init - 1) {
        //                         if (time_vector[i].time != time_vector[i - 1].time) {
        //                             file1 << time_vector[i].time;
        //                         }
        //                     }
        //                 }
        //                 #endif
        //                 file1.close();

        //             // SIMULATION OUTPUT
        //                 relavtive_pth = SIM_OUTPUT_relPATH;
        //                 relavtive_pth.append("Mean_PDFs_");
        //                 relavtive_pth.append(temp_str);
        //                 relavtive_pth.append(".bin");

        //                 std::ofstream myfile(relavtive_pth, std::ios::out | std::ios::binary);
        //                 assert (myfile.is_open());

        //                 myfile.write((char*)&storeFrames[(k * max_frames_file + frames_init) * Problem_Domain.Total_Nodes()], sizeof(float) * frames_in_file * Problem_Domain.Total_Nodes());
        //                 myfile.close();
        //                 std::cout << "Simulation output file " << k << " completed!\n";
                        
        //             }
        //         }

        //     saving_active = false;
        // }
//     }

// }



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// The following functions are NOT inside the Simulation Data information

template<uint16_t PHASE_SPACE_DIM, uint16_t PARAM_SPACE_DIM, class T> 
__device__ 
inline gridPoint<PHASE_SPACE_DIM, T> VECTOR_FIELD(gridPoint<PHASE_SPACE_DIM, T>          X, 
                                                 double          t, 
                                                                    const Param_vec<PARAM_SPACE_DIM, T>   parameter, 
                                                                                                    const UINT      mode, 
                                                                                                    const double    extra_param[]) {

	return { VEC_FIELD };
}
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
template<uint16_t PHASE_SPACE_DIM, uint16_t PARAM_SPACE_DIM, class T>
__device__
inline T DIVERGENCE_FIELD (gridPoint<PHASE_SPACE_DIM, T>      X,
                                                    T       t, 
                             const Param_vec<PARAM_SPACE_DIM, T>      parameter, 
                                                                const UINT      mode, 
                                                                const double    extra_param[]) {

	return DIVERGENCE;
}

#endif