#include <Sim_data.cuh>

LogFrames::LogFrames() {
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
};

LogSimulation::LogSimulation(uintType size) {
    LogFrames.resize(size);
};

void LogSimulation::resize(uintType size) {
    LogFrames.resize(size);
};

int16_t LogSimulation::writeSimulationLog_toFile(const std::string& fileName, const std::string fileExtension, const std::string_view fileRelativePath) {

    std::string fileCompleteInfo{fileRelativePath};
    fileCompleteInfo += "/output/";
    fileCompleteInfo += fileName;
    fileCompleteInfo += fileExtension;

    std::cout << "[INFO] Saving log file into " + fileCompleteInfo << std::endl;

    std::ofstream logFile(fileCompleteInfo, std::ios::out);
    if (!logFile.is_open()) {
        std::cout << termcolor::bold << termcolor::yellow << "[WARNING] Log file cannot be opened. Log information will not be written." << std::endl;
        std::cout << termcolor::reset;
        return -1;
    }

    // We follow the following order: Timings >> Iterations/Particles

    logFile << "Time in Sim, Iteration index in Sim, Time [s]: AMR, Time [s]: Interpolation, Time[s]: Advection, Time[s]: Reinitialization, , Relevant Particles (AMR), Conj.Grad. Iterations, Total particles (Advection)\n";

    for (auto& LogFrame : LogFrames) {

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
};
