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
    plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::info, &consoleAppender);

    cuda_filter::InputArgsParser parser(argc, argv);
    cuda_filter::FilterOptions options = parser.parseArgs();

    cuda_filter::InputHandler inputHandler(options);
    if (!inputHandler.isOpened())
    {
        PLOG_ERROR << "Failed to initialize input source";
        return -1;
    }

    cv::Mat blurKernel = cuda_filter::FilterUtils::createFilterKernel(
        cuda_filter::FilterType::BLUR, 3, 1.0f);

    cv::Mat sharpenKernel = cuda_filter::FilterUtils::createFilterKernel(
        cuda_filter::FilterType::SHARPEN, 3, 1.0f);

    PLOG_INFO << "Filter pipeline enabled: blur -> sharpen";
    PLOG_INFO << "Wipe transition enabled";
    PLOG_INFO << "Press 'ESC' to exit";

    cv::Mat frame;
    cv::Mat stage1;
    cv::Mat stage2;
    cv::Mat transitionOutput;

    double fps = 0.0;
    int frameCount = 0;
    double startTime = static_cast<double>(cv::getTickCount());

    float transition = 0.0f;

    while (true)
    {
        if (!inputHandler.readFrame(frame))
        {
            PLOG_ERROR << "Failed to read frame";
            break;
        }

        const double pipelineStart = static_cast<double>(cv::getTickCount());

        // Stage 1: blur
        cuda_filter::applyFilterGPU(frame, stage1, blurKernel);

        // Stage 2: sharpen
        cuda_filter::applyFilterGPU(stage1, stage2, sharpenKernel);

        const double pipelineEnd = static_cast<double>(cv::getTickCount());
        const double pipelineTime =
            (pipelineEnd - pipelineStart) / cv::getTickFrequency();

        // Wipe transition from original frame to pipeline result
        transitionOutput.create(frame.size(), frame.type());

        int wipeX = static_cast<int>(frame.cols * transition);

        for (int y = 0; y < frame.rows; y++)
        {
            for (int x = 0; x < frame.cols; x++)
            {
                if (x < wipeX)
                    transitionOutput.at<cv::Vec3b>(y, x) = stage2.at<cv::Vec3b>(y, x);
                else
                    transitionOutput.at<cv::Vec3b>(y, x) = frame.at<cv::Vec3b>(y, x);
            }
        }

        transition += 0.01f;
        if (transition > 1.0f)
            transition = 0.0f;

        frameCount++;
        double now = static_cast<double>(cv::getTickCount());
        if ((now - startTime) / cv::getTickFrequency() >= 1.0)
        {
            fps = frameCount;
            frameCount = 0;
            startTime = now;
        }

        std::string text = "Pipeline FPS: " + std::to_string(static_cast<int>(fps)) +
                           " Time: " + std::to_string(pipelineTime * 1000).substr(0, 5) + "ms";

        cv::putText(transitionOutput, text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 0), 2);

        cv::putText(transitionOutput, "Pipeline: blur -> sharpen",
                    cv::Point(10, transitionOutput.rows - 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 0), 2);

        cv::putText(transitionOutput, "Wipe transition",
                    cv::Point(10, transitionOutput.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 0), 2);

        inputHandler.displayFrame(transitionOutput);

        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }

    PLOG_INFO << "Application terminated";
    return 0;
}