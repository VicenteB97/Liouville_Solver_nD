#ifndef __SIM_DATA_CUH__
#define __SIM_DATA_CUH__

#include "config.hpp"
#include "utils/numeric_defs.cuh"
#include "headers.cuh"
#include "indicators/cursor_control.hpp"
#include "indicators/progress_bar.hpp"

// Simulation log for further analysis
class LogFrames {
public:
    // Attributes:
    double simTime;
    uintType simIteration;

    double log_AMR_Time;
    uintType log_AMR_RelevantParticles;
    std::string log_AMR_Message;

    double log_Interpolation_Time;
    uint16_t log_Interpolation_Iterations;
    std::string log_Interpolation_Message;

    double log_Advection_Time;
    uintType log_Advection_TotalParticles;
    std::string log_Advection_Message;

    double log_Reinitialization_Time;
    std::string log_Reinitialization_Message;

    uint64_t log_MemoryUsage;

    double log_TotalFrameTime;
    std::string log_TotalFrameTime_Message;

    uint16_t log_MessageLevel; // 0 for info, 1 for warning, 2 for error

public:
    // Constructor
    LogFrames();

};



// The wrapper class
class LogSimulation {
public:
    std::vector<LogFrames> LogFrames;

public:
    LogSimulation(uintType size = 1);

public:
    void resize(uintType size = 1);

    int16_t writeSimulationLog_toFile(const std::string& fileName = "Simulation Log File", const std::string fileExtension = ".csv", const std::string_view fileRelativePath = SRC_DIR);
};

#endif