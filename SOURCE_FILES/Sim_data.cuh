#ifndef __SIM_DATA_CUH__
#define __SIM_DATA_CUH__

#include "Domain.cuh"
#include "Probability.cuh"

// Simulation log for further analysis
class Logger {
public:
    std::vector<double> subFrame_time;
    std::vector<int32_t> ConvergenceIterations;

    // default constructor
    Logger(const uint32_t& size = 1) {
        subFrame_time.resize(5 * size, (double)0);
        ConvergenceIterations.resize(size, (int32_t)0);
    }

public:
    inline void resize(const int32_t& size) {
        subFrame_time.resize(5 * size);
        ConvergenceIterations.resize(size);
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

#endif