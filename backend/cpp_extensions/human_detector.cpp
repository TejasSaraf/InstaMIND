#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

static constexpr int NET_INPUT_SIZE = 416;
static constexpr float CONF_THRESHOLD = 0.45f;
static constexpr float NMS_THRESHOLD = 0.40f;
static constexpr int PERSON_CLASS = 0;
static constexpr int FRAME_SKIP = 1;

static const cv::Scalar BOX_COLOR = cv::Scalar(0, 255, 0);
static const cv::Scalar TEXT_BG = cv::Scalar(0, 0, 0);
static const cv::Scalar TEXT_COLOR = cv::Scalar(255, 255, 255);

static std::string resolve_model_path(const std::string &filename)
{

    if (fs::exists(filename))
        return filename;

    std::string rel = "../models/" + filename;
    if (fs::exists(rel))
        return rel;

    auto exe_dir = fs::canonical("/proc/self/exe").parent_path();
    std::string beside = (exe_dir / filename).string();
    if (fs::exists(beside))
        return beside;

    return rel;
}

static cv::dnn::Net load_yolo(const std::string &cfg_path,
                              const std::string &weights_path)
{
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(cfg_path, weights_path);
    if (net.empty())
    {
        throw std::runtime_error("Failed to load YOLO model from:\n  cfg: " + cfg_path + "\n  weights: " + weights_path);
    }

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
#ifdef __APPLE__
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

#else
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#endif

    return net;
}

struct Detection
{
    cv::Rect box;
    float confidence;
};

static std::vector<Detection> detect_humans(cv::dnn::Net &net,
                                            const cv::Mat &frame,
                                            float conf_thresh,
                                            float nms_thresh)
{

    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0,
                           cv::Size(NET_INPUT_SIZE, NET_INPUT_SIZE),
                           cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::String> out_names = net.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outs;
    net.forward(outs, out_names);

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    const int fw = frame.cols;
    const int fh = frame.rows;

    for (const auto &out : outs)
    {
        const auto *data = reinterpret_cast<const float *>(out.data);
        for (int j = 0; j < out.rows; ++j, data += out.cols)
        {

            float obj_conf = data[4];
            if (obj_conf < conf_thresh)
                continue;

            float person_score = data[5 + PERSON_CLASS] * obj_conf;
            if (person_score < conf_thresh)
                continue;

            int cx = static_cast<int>(data[0] * fw);
            int cy = static_cast<int>(data[1] * fh);
            int w = static_cast<int>(data[2] * fw);
            int h = static_cast<int>(data[3] * fh);
            int x = cx - w / 2;
            int y = cy - h / 2;

            boxes.emplace_back(x, y, w, h);
            confidences.push_back(person_score);
            class_ids.push_back(PERSON_CLASS);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thresh, nms_thresh, indices);

    std::vector<Detection> results;
    results.reserve(indices.size());
    for (int idx : indices)
    {
        results.push_back({boxes[idx], confidences[idx]});
    }
    return results;
}

static void draw_detections(cv::Mat &frame,
                            const std::vector<Detection> &dets,
                            int frame_idx, double fps)
{
    for (const auto &d : dets)
    {

        cv::Rect safe(
            std::max(d.box.x, 0),
            std::max(d.box.y, 0),
            std::min(d.box.width, frame.cols - std::max(d.box.x, 0)),
            std::min(d.box.height, frame.rows - std::max(d.box.y, 0)));

        cv::rectangle(frame, safe, BOX_COLOR, 2);

        char label[64];
        std::snprintf(label, sizeof(label), "person %.0f%%", d.confidence * 100.0f);
        int baseline = 0;
        cv::Size tsz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(frame,
                      cv::Point(safe.x, safe.y - tsz.height - 6),
                      cv::Point(safe.x + tsz.width + 4, safe.y),
                      TEXT_BG, cv::FILLED);
        cv::putText(frame, label,
                    cv::Point(safe.x + 2, safe.y - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1);
    }

    char info[64];
    std::snprintf(info, sizeof(info), "Frame %d | %.1f FPS", frame_idx, fps);
    cv::putText(frame, info, cv::Point(8, 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
}

static void print_json(int frame_idx, double timestamp_s,
                       const std::vector<Detection> &dets)
{
    std::cout << "{\"frame\":" << frame_idx
              << ",\"time\":" << std::round(timestamp_s * 100.0) / 100.0
              << ",\"persons\":" << dets.size()
              << ",\"boxes\":[";
    for (size_t i = 0; i < dets.size(); ++i)
    {
        const auto &b = dets[i].box;
        if (i > 0)
            std::cout << ',';
        std::cout << "{\"x\":" << b.x << ",\"y\":" << b.y
                  << ",\"w\":" << b.width << ",\"h\":" << b.height
                  << ",\"conf\":" << std::round(dets[i].confidence * 1000.0f) / 1000.0f
                  << '}';
    }
    std::cout << "]}" << std::endl;
}

static void print_usage(const char *prog)
{
    std::cerr << "Usage: " << prog
              << " <input_video> [output_video] [--skip N] [--conf 0.5] [--nms 0.4] [--show]\n"
              << "\nOptions:\n"
              << "  --skip  N     Process every Nth frame (default: 1)\n"
              << "  --conf  F     Confidence threshold (default: 0.45)\n"
              << "  --nms   F     NMS IoU threshold (default: 0.40)\n"
              << "  --show        Display live preview window\n"
              << "\nRequires yolov4-tiny.cfg and yolov4-tiny.weights in ../models/\n";
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_path;
    std::string output_path;
    int skip = FRAME_SKIP;
    float conf = CONF_THRESHOLD;
    float nms = NMS_THRESHOLD;
    bool show_gui = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--skip" && i + 1 < argc)
        {
            skip = std::stoi(argv[++i]);
        }
        else if (arg == "--conf" && i + 1 < argc)
        {
            conf = std::stof(argv[++i]);
        }
        else if (arg == "--nms" && i + 1 < argc)
        {
            nms = std::stof(argv[++i]);
        }
        else if (arg == "--show")
        {
            show_gui = true;
        }
        else if (arg[0] == '-')
        {
            print_usage(argv[0]);
            return 1;
        }
        else if (input_path.empty())
        {
            input_path = arg;
        }
        else if (output_path.empty())
        {
            output_path = arg;
        }
    }

    if (input_path.empty())
    {
        print_usage(argv[0]);
        return 1;
    }
    if (output_path.empty())
    {
        auto p = fs::path(input_path);
        output_path = (p.parent_path() / (p.stem().string() + "_detected" + p.extension().string())).string();
    }

    std::string cfg_path = resolve_model_path("yolov4-tiny.cfg");
    std::string weights_path = resolve_model_path("yolov4-tiny.weights");

    std::cerr << "[human_detector] Loading YOLO model...\n";
    std::cerr << "  cfg:     " << cfg_path << "\n";
    std::cerr << "  weights: " << weights_path << "\n";

    cv::dnn::Net net;
    try
    {
        net = load_yolo(cfg_path, weights_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << "\n";
        std::cerr << "\nDownload the model files:\n"
                  << "  wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights -P ../models/\n"
                  << "  wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg -P ../models/\n";
        return 1;
    }

    cv::VideoCapture cap(input_path, cv::CAP_FFMPEG);
    if (!cap.isOpened())
    {
        cap.open(input_path);
    }
    if (!cap.isOpened())
    {
        std::cerr << "[ERROR] Cannot open video: " << input_path << "\n";
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0)
        fps = 30.0;
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double duration_s = total_frames / fps;

    std::cerr << "[human_detector] Video: " << frame_w << "x" << frame_h
              << " @ " << fps << " fps, " << total_frames << " frames ("
              << std::round(duration_s * 10.0) / 10.0 << "s)\n";
    std::cerr << "[human_detector] Config: skip=" << skip
              << " conf=" << conf << " nms=" << nms << "\n";

    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(output_path, fourcc, fps, cv::Size(frame_w, frame_h));
    if (!writer.isOpened())
    {
        std::cerr << "[WARN] Cannot write to " << output_path << ", continuing without output file.\n";
    }

    int frame_idx = 0;
    int processed = 0;
    int total_persons = 0;
    double total_infer_ms = 0.0;

    std::vector<Detection> last_dets;

    std::cerr << "[human_detector] Processing...\n";

    cv::Mat frame;
    while (cap.read(frame))
    {
        double timestamp_s = frame_idx / fps;

        if (frame_idx % skip == 0)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            last_dets = detect_humans(net, frame, conf, nms);
            auto t1 = std::chrono::high_resolution_clock::now();

            double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_infer_ms += infer_ms;
            processed++;
            total_persons += static_cast<int>(last_dets.size());

            double running_fps = (infer_ms > 0) ? (1000.0 / infer_ms) : 0;
            print_json(frame_idx, timestamp_s, last_dets);

            draw_detections(frame, last_dets, frame_idx, running_fps);
        }
        else
        {

            double running_fps = (total_infer_ms > 0)
                                     ? (processed * 1000.0 / total_infer_ms)
                                     : 0;
            draw_detections(frame, last_dets, frame_idx, running_fps);
        }

        if (writer.isOpened())
            writer.write(frame);

        if (show_gui)
        {
            cv::imshow("instaMIND — Human Detector", frame);
            if (cv::waitKey(1) == 27)
                break;
        }

        frame_idx++;
    }

    cap.release();
    writer.release();
    if (show_gui)
        cv::destroyAllWindows();

    double avg_ms = (processed > 0) ? (total_infer_ms / processed) : 0;
    std::cerr << "\n[human_detector] Done.\n"
              << "  Frames processed:  " << processed << " / " << frame_idx << "\n"
              << "  Total persons:     " << total_persons << "\n"
              << "  Avg inference:     " << std::round(avg_ms * 10.0) / 10.0 << " ms\n"
              << "  Avg detection FPS: " << std::round(1000.0 / std::max(avg_ms, 0.1) * 10.0) / 10.0 << "\n"
              << "  Output:            " << output_path << "\n";

    return 0;
}
