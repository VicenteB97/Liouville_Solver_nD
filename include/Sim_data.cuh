#ifndef __SIM_DATA_CUH__
#define __SIM_DATA_CUH__

#include "utils/paths.cuh"
#include "utils/numeric_defs.cuh"
#include "headers.cuh"
#include "indicators/cursor_control.hpp"
#include "indicators/progress_bar.hpp"

// Simulation log for further analysis
class LogFrames {
public:
    // Attributes:
    double simTime;
    UINT simIteration;

    double log_AMR_Time;
    UINT log_AMR_RelevantParticles;
    std::string log_AMR_Message;

    double log_Interpolation_Time;
    uint16_t log_Interpolation_Iterations;
    std::string log_Interpolation_Message;

    double log_Advection_Time;
    UINT log_Advection_TotalParticles;
    std::string log_Advection_Message;

    double log_Reinitialization_Time;
    std::string log_Reinitialization_Message;

    uint64_t log_MemoryUsage;

    double log_TotalFrameTime;
    std::string log_TotalFrameTime_Message;

    uint16_t log_MessageLevel; // 0 for info, 1 for warning, 2 for error

public:
    // Constructor
    LogFrames(){
        log_AMR_Time = 0;
        log_AMR_RelevantParticles = 0;
        log_AMR_Message = "";

        log_Interpolation_Time = 0;
        log_Interpolation_Iterations = 0;
        log_Interpolation_Message = "";

        log_Advection_Time = 0;
        log_Advection_TotalParticles = 0;
        log_Advection_Message = "";

        log_MemoryUsage = 0;

        log_Reinitialization_Time = 0;
        log_Reinitialization_Message = "";

        log_MessageLevel = 0;
    }
    
    // Destructor
    ~LogFrames(){};

};



// The wrapper class
class LogSimulation {
public:
    std::vector<LogFrames> LogFrames;

public:
    LogSimulation(UINT size = 1){
        LogFrames.resize(size);
    };

    ~LogSimulation(){}

public:

    void resize(UINT size = 1){
        LogFrames.resize(size);
    }


    int16_t writeSimulationLog_toFile(const std::string& fileName = "Simulation Log File", const std::string fileExtension = ".csv", const std::string fileRelativePath = LOG_OUTPUT_relPATH){

        const std::string fileCompleteInfo = fileRelativePath + fileName + fileExtension;

        std::cout << "[INFO] Saving log file into " + fileCompleteInfo << std::endl;

        std::ofstream logFile(fileCompleteInfo, std::ios::out);
        if(!logFile.is_open()){
            std::cout << termcolor::bold << termcolor::yellow << "[WARNING] Log file cannot be opened. Log information will not be written." << std::endl;
            std::cout << termcolor::reset;
            return -1;
        }

        // We follow the following order: Timings >> Iterations/Particles

        logFile << "Time in Sim, Iteration index in Sim, Time [s]: AMR, Time [s]: Interpolation, Time[s]: Advection, Time[s]: Reinitialization, , Relevant Particles (AMR), Conj.Grad. Iterations, Total particles (Advection)\n"; 

        for(auto &LogFrame : LogFrames){

            std::string inputRow = "";

            inputRow = std::to_string(LogFrame.simTime) + "," + std::to_string(LogFrame.simIteration) + ","
                        + std::to_string(LogFrame.log_AMR_Time) + "," + std::to_string(LogFrame.log_Interpolation_Time) + ","
                        + std::to_string(LogFrame.log_Advection_Time) + "," + std::to_string(LogFrame.log_Reinitialization_Time) + ","
                        + " ," + std::to_string(LogFrame.log_AMR_RelevantParticles) + "," + std::to_string(LogFrame.log_Interpolation_Iterations) + ","
                        + std::to_string(LogFrame.log_Advection_TotalParticles) + "\n";

            logFile << inputRow;
        }

        logFile.close();
        return 0;
    }


};

#endif