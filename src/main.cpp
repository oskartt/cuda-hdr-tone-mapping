#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>
#include "input_args_parser/input_args_parser.h"
#include "utils/input_handler.h"
#include "utils/filter_utils.h"
#include "kernels/kernels.h"

int main(int argc, char **argv)
{
    // Initialize logger
    plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::info, &consoleAppender);

    // Read command-line options such as filter type and kernel size
    cuda_filter::InputArgsParser parser(argc, argv);
    cuda_filter::FilterOptions options = parser.parseArgs();

    // Open webcam or input source
    cuda_filter::InputHandler inputHandler(options);
    if (!inputHandler.isOpened())
    {
        PLOG_ERROR << "Failed to initialize input source";
        return -1;
    }

    // Convert filter name, for example "hdr", into FilterType enum
    cuda_filter::FilterType filterType =
        cuda_filter::FilterUtils::stringToFilterType(options.filterType);

    // Create normal convolution kernel for standard filters.
    // HDR uses a separate CUDA kernel, so this is only a placeholder for HDR.
    cv::Mat kernel = cuda_filter::FilterUtils::createFilterKernel(
        filterType, options.kernelSize, options.intensity);

    // HDR parameters used by the CUDA tone mapping kernel
    float exposure = 1.5f;
    float gamma = 2.2f;

    PLOG_INFO << "Filter: " << options.filterType;

    if (filterType == cuda_filter::FilterType::HDR_TONEMAPPING)
    {
        PLOG_INFO << "HDR tone mapping enabled";
        PLOG_INFO << "Exposure: " << exposure << ", Gamma: " << gamma;
    }

    cv::Mat frame, filteredCPU, filteredGPU;

    double fpsCPU = 0.0, fpsGPU = 0.0;
    int frameCountCPU = 0, frameCountGPU = 0;

    double startTimeCPU = static_cast<double>(cv::getTickCount());
    double startTimeGPU = static_cast<double>(cv::getTickCount());

    PLOG_INFO << "Press 'ESC' to exit";

    while (true)
    {
        // Capture a frame from webcam or input source
        if (!inputHandler.readFrame(frame))
        {
            PLOG_ERROR << "Failed to read frame";
            break;
        }

        // CPU side is used for comparison.
        // For HDR, the CPU side shows the original image.
        const double cpuStart = static_cast<double>(cv::getTickCount());

        if (filterType == cuda_filter::FilterType::HDR_TONEMAPPING)
        {
            frame.copyTo(filteredCPU);
        }
        else
        {
            cuda_filter::applyFilterCPU(frame, filteredCPU, kernel);
        }

        const double cpuEnd = static_cast<double>(cv::getTickCount());
        const double cpuTime = (cpuEnd - cpuStart) / cv::getTickFrequency();

        frameCountCPU++;
        if ((cpuEnd - startTimeCPU) / cv::getTickFrequency() >= 1.0)
        {
            fpsCPU = frameCountCPU;
            frameCountCPU = 0;
            startTimeCPU = cpuEnd;
        }

        // GPU processing section.
        // If HDR is selected, run the CUDA HDR tone mapping kernel.
        const double gpuStart = static_cast<double>(cv::getTickCount());

        if (filterType == cuda_filter::FilterType::HDR_TONEMAPPING)
        {
            cuda_filter::applyHDRGPU(frame, filteredGPU, exposure, gamma);
        }
        else
        {
            cuda_filter::applyFilterGPU(frame, filteredGPU, kernel);
        }

        const double gpuEnd = static_cast<double>(cv::getTickCount());
        const double gpuTime = (gpuEnd - gpuStart) / cv::getTickFrequency();

        frameCountGPU++;
        if ((gpuEnd - startTimeGPU) / cv::getTickFrequency() >= 1.0)
        {
            fpsGPU = frameCountGPU;
            frameCountGPU = 0;
            startTimeGPU = gpuEnd;
        }

        // Text overlay showing CPU and GPU timing
        std::string cpuText = "CPU FPS: " + std::to_string(static_cast<int>(fpsCPU)) +
                              " Time: " + std::to_string(cpuTime * 1000).substr(0, 4) + "ms";

        std::string gpuText = "GPU FPS: " + std::to_string(static_cast<int>(fpsGPU)) +
                              " Time: " + std::to_string(gpuTime * 1000).substr(0, 4) + "ms";

        cv::putText(filteredCPU, cpuText, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 0), 2);

        cv::putText(filteredGPU, gpuText, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 0), 2);

        // Combine original/CPU and GPU HDR output side by side
        cv::Mat combined;
        cv::hconcat(filteredCPU, filteredGPU, combined);

        cv::putText(combined, "Original / CPU", cv::Point(10, combined.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 0), 2);

        cv::putText(combined, "GPU HDR / Filter", cv::Point(combined.cols / 2 + 10, combined.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 0), 2);

        // Display the final comparison frame
        if (options.preview)
            inputHandler.displaySideBySide(frame, combined);
        else
            inputHandler.displayFrame(combined);

        // ESC exits the program
        if (cv::waitKey(1) == 27)
            break;
    }

    PLOG_INFO << "Application terminated";
    return 0;
}
