#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <numeric>
#include <algorithm>

namespace py = pybind11;

static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const unsigned char *buf, unsigned int bufLen)
{
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (bufLen--)
    {
        char_array_3[i++] = *(buf++);
        if (i == 3)
        {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            for (i = 0; (i < 4); i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }
    if (i)
    {
        for (j = i; j < 3; j++)
            char_array_3[j] = '\0';
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;
        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];
        while ((i++ < 3))
            ret += '=';
    }
    return ret;
}

struct KeyFrame
{
    double timestamp_seconds;
    std::string image_b64;
    int frame_index;
};

void init_keyframe(py::module &m)
{
    py::class_<KeyFrame>(m, "KeyFrame")
        .def(py::init<double, std::string, int>())
        .def_readwrite("timestamp_seconds", &KeyFrame::timestamp_seconds)
        .def_readwrite("image_b64", &KeyFrame::image_b64)
        .def_readwrite("frame_index", &KeyFrame::frame_index);
}

int get_budget(double duration)
{
    if (duration < 3.0)
        return 3;
    if (duration < 8.0)
        return 4;
    if (duration < 20.0)
        return 6;
    return 8;
}

std::vector<std::pair<int, double>> pick_uniform(const std::vector<std::pair<int, double>> &pairs, int n)
{
    int m = pairs.size();
    if (m <= n)
        return pairs;
    std::vector<std::pair<int, double>> chosen;
    for (int i = 0; i < n; ++i)
    {
        int idx = std::round(static_cast<double>(i) * (m - 1) / (n - 1));
        chosen.push_back(pairs[idx]);
    }
    return chosen;
}

std::vector<std::vector<KeyFrame>> sample_keyframes_cpp(
    const std::string &video_path,
    int target_n,
    double short_clip_threshold_s,
    double window_s,
    double window_overlap_s,
    double long_clip_threshold_s,
    int encode_quality)
{
    cv::VideoCapture cap(video_path, cv::CAP_FFMPEG);
    if (!cap.isOpened())
    {
        cap.open(video_path);
        if (!cap.isOpened())
            return {};
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0)
        fps = 30.0;

    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    if (total_frames <= 0)
    {
        int fi = 0;
        while (cap.grab())
        {
            fi++;
        }
        total_frames = fi;
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    }

    if (total_frames <= 0)
        return {};

    double duration_s = total_frames / fps;
    std::vector<std::pair<int, double>> all_indices;
    all_indices.reserve(total_frames);
    for (int i = 0; i < total_frames; ++i)
    {
        all_indices.push_back({i, i / fps});
    }

    std::vector<std::vector<std::pair<int, double>>> windows_indices;
    if (duration_s <= long_clip_threshold_s)
    {
        int budget = get_budget(duration_s);
        windows_indices.push_back(pick_uniform(all_indices, budget));
    }
    else
    {
        double step_s = window_s - window_overlap_s;
        int budget = get_budget(window_s);
        double t_start = 0.0;
        while (t_start < duration_s)
        {
            double t_end = t_start + window_s;
            std::vector<std::pair<int, double>> window_pairs;
            for (const auto &pair : all_indices)
            {
                if (pair.second >= t_start && pair.second < t_end)
                {
                    window_pairs.push_back(pair);
                }
            }
            if (!window_pairs.empty())
            {
                windows_indices.push_back(pick_uniform(window_pairs, budget));
            }
            t_start += step_s;
        }
    }

    std::vector<std::vector<KeyFrame>> result;
    std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, encode_quality};

    for (const auto &w : windows_indices)
    {
        std::vector<KeyFrame> window_frames;
        for (const auto &p : w)
        {
            cap.set(cv::CAP_PROP_POS_FRAMES, p.first);
            cv::Mat frame;
            if (cap.read(frame))
            {
                std::vector<uchar> buf;
                cv::imencode(".jpg", frame, buf, encode_params);
                std::string b64 = base64_encode(buf.data(), buf.size());

                KeyFrame kf;
                kf.timestamp_seconds = std::round(p.second * 1000.0) / 1000.0;
                kf.image_b64 = b64;
                kf.frame_index = p.first;
                window_frames.push_back(kf);
            }
        }
        if (!window_frames.empty())
        {
            result.push_back(window_frames);
        }
    }

    cap.release();
    return result;
}

py::dict analyze_video_cpp(const std::string &video_path, int max_frames, double target_ms)
{
    cv::VideoCapture cap(video_path, cv::CAP_FFMPEG);
    if (!cap.isOpened())
    {
        cap.open(video_path);
        if (!cap.isOpened())
            throw std::runtime_error("Unable to open video");
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0)
        fps = 30.0;
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    double duration_seconds = (fps > 0) ? (total_frames / fps) : 0.0;

    std::vector<float> motion_scores;
    std::vector<float> brightness_scores;
    std::vector<float> horizontal_scores;
    std::vector<float> area_changes;
    std::vector<float> frame_latencies_ms;

    cv::Mat prev_gray;
    float prev_area = 0.0f;
    double downscale = 0.5;
    int skip_stride = 1;
    int frame_idx = 0;
    int processed_count = 0;

    while (processed_count < max_frames)
    {
        cv::Mat frame;
        if (!cap.read(frame))
            break;

        if (frame_idx % skip_stride != 0)
        {
            frame_idx++;
            continue;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        cv::Mat small, gray, blur;
        cv::resize(frame, small, cv::Size(), downscale, downscale, cv::INTER_AREA);
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);

        cv::Scalar mean_brightness = cv::mean(gray);
        brightness_scores.push_back(static_cast<float>(mean_brightness[0]));

        if (prev_gray.empty())
        {
            motion_scores.push_back(0.0f);
        }
        else
        {
            cv::Mat diff;
            cv::absdiff(gray, prev_gray, diff);
            cv::Scalar mean_diff = cv::mean(diff);
            motion_scores.push_back(static_cast<float>(mean_diff[0]));
        }

        cv::Mat th;
        cv::threshold(blur, th, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(th, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        float aspect_ratio = 0.0f;
        float area = 0.0f;
        if (!contours.empty())
        {
            auto largest_contour = std::max_element(contours.begin(), contours.end(),
                                                    [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b)
                                                    {
                                                        return cv::contourArea(a) < cv::contourArea(b);
                                                    });
            cv::Rect rect = cv::boundingRect(*largest_contour);
            aspect_ratio = static_cast<float>(rect.width) / std::max(static_cast<float>(rect.height), 1.0f);
            area = static_cast<float>(rect.width * rect.height);
        }

        float horizontal_prob = 1.0f / (1.0f + std::exp(-(aspect_ratio - 1.4f) * 3.0f));
        horizontal_scores.push_back(horizontal_prob);
        area_changes.push_back(std::abs(area - prev_area));

        prev_area = area;
        prev_gray = gray;

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        float elapsed_ms = elapsed.count();
        frame_latencies_ms.push_back(elapsed_ms);

        if (elapsed_ms > target_ms)
        {
            downscale = std::max(0.25, downscale - 0.1);
            skip_stride = std::min(4, skip_stride + 1);
        }

        processed_count++;
        frame_idx++;
    }

    cap.release();

    if (frame_latencies_ms.empty())
    {
        throw std::runtime_error("No frames processed from uploaded video.");
    }

    auto calc_mean = [](const std::vector<float> &v)
    {
        if (v.empty())
            return 0.0f;
        float sum = std::accumulate(v.begin(), v.end(), 0.0f);
        return sum / v.size();
    };

    auto calc_std = [](const std::vector<float> &v, float mean)
    {
        if (v.size() <= 1)
            return 0.0f;
        float sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0f);
        float stdev = std::sqrt(std::max(0.0f, (sq_sum / v.size()) - (mean * mean)));
        return stdev;
    };

    auto calc_percentile = [](std::vector<float> v, float p)
    {
        if (v.empty())
            return 0.0f;
        size_t idx = static_cast<size_t>(std::round(p * (v.size() - 1)));
        std::nth_element(v.begin(), v.begin() + idx, v.end());
        return v[idx];
    };

    float motion_mean = calc_mean(motion_scores);
    float motion_std = calc_std(motion_scores, motion_mean);
    float bright_mean = calc_mean(brightness_scores);
    float horizontal_mean = calc_mean(horizontal_scores);
    float area_delta_mean = calc_mean(area_changes);

    int violations = 0;
    float max_ms = 0.0f;
    for (float ms : frame_latencies_ms)
    {
        if (ms > target_ms)
            violations++;
        if (ms > max_ms)
            max_ms = ms;
    }

    py::list motion_series;
    for (size_t i = 0; i < std::min<size_t>(120, motion_scores.size()); i++)
    {
        motion_series.append(std::round(motion_scores[i] * 10000.0) / 10000.0);
    }

    py::list horizontal_series;
    for (size_t i = 0; i < std::min<size_t>(120, horizontal_scores.size()); i++)
    {
        horizontal_series.append(std::round(horizontal_scores[i] * 10000.0) / 10000.0);
    }

    py::dict latency_summary;
    latency_summary["target_ms"] = static_cast<int>(target_ms);
    latency_summary["frame_count_processed"] = static_cast<int>(frame_latencies_ms.size());
    latency_summary["p50_ms"] = calc_percentile(frame_latencies_ms, 0.5f);
    latency_summary["p95_ms"] = calc_percentile(frame_latencies_ms, 0.95f);
    latency_summary["max_ms"] = max_ms;
    latency_summary["violations"] = violations;
    latency_summary["met_target"] = (max_ms <= target_ms);
    latency_summary["downscale_final"] = downscale;
    latency_summary["skip_stride_final"] = skip_stride;

    py::dict video_signals;
    video_signals["fps"] = fps;
    video_signals["total_frames"] = total_frames;
    video_signals["duration_seconds"] = duration_seconds;
    video_signals["sample_count"] = static_cast<int>(frame_latencies_ms.size());
    video_signals["brightness_mean"] = bright_mean;
    video_signals["motion_mean"] = motion_mean;
    video_signals["motion_std"] = motion_std;
    video_signals["motion_series"] = motion_series;

    py::dict pose_signals;
    pose_signals["pose_sample_count"] = static_cast<int>(horizontal_scores.size());
    pose_signals["horizontal_posture_score"] = horizontal_mean;
    pose_signals["area_change_mean"] = area_delta_mean;
    pose_signals["horizontal_series"] = horizontal_series;

    py::dict result;
    result["video"] = video_signals;
    result["pose"] = pose_signals;
    result["latency"] = latency_summary;

    return result;
}

std::vector<KeyFrame> extract_frames_at_interval_cpp(
    const std::string &video_path,
    double interval_seconds,
    int encode_quality)
{
    cv::VideoCapture cap(video_path, cv::CAP_FFMPEG);
    if (!cap.isOpened())
    {
        cap.open(video_path);
        if (!cap.isOpened())
            return {};
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0)
        fps = 30.0;
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (total_frames <= 0)
    {
        cap.release();
        return {};
    }

    double duration_s = total_frames / fps;
    std::vector<KeyFrame> result;
    std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, encode_quality};

    for (double t = 0.0; t < duration_s; t += interval_seconds)
    {
        int frame_idx = static_cast<int>(t * fps);
        if (frame_idx >= total_frames)
            break;
        cap.set(cv::CAP_PROP_POS_FRAMES, frame_idx);
        cv::Mat frame;
        if (cap.read(frame))
        {
            int max_dim = std::max(frame.cols, frame.rows);
            if (max_dim > 896)
            {
                double scale = 896.0 / max_dim;
                cv::resize(frame, frame, cv::Size(), scale, scale, cv::INTER_AREA);
            }
            std::vector<uchar> buf;
            cv::imencode(".jpg", frame, buf, encode_params);
            std::string b64 = base64_encode(buf.data(), buf.size());

            KeyFrame kf;
            kf.timestamp_seconds = std::round(t * 1000.0) / 1000.0;
            kf.image_b64 = b64;
            kf.frame_index = frame_idx;
            result.push_back(kf);
        }
    }

    cap.release();
    return result;
}

PYBIND11_MODULE(video_analyzer_cpp, m)
{
    m.doc() = "Fast C++ Video Analyzer and Frame Extractor";
    init_keyframe(m);
    m.def("sample_keyframes", &sample_keyframes_cpp, "Extract keyframes using fast-seek",
          py::arg("video_path"),
          py::arg("target_n") = 6,
          py::arg("short_clip_threshold_s") = 3.0,
          py::arg("window_s") = 5.0,
          py::arg("window_overlap_s") = 1.0,
          py::arg("long_clip_threshold_s") = 30.0,
          py::arg("encode_quality") = 85);
    m.def("analyze", &analyze_video_cpp, "Analyze video for motion and posture signals",
          py::arg("video_path"),
          py::arg("max_frames") = 600,
          py::arg("target_ms") = 100.0);
    m.def("extract_frames_at_interval", &extract_frames_at_interval_cpp,
          "Extract one frame every N seconds (lightweight, for vision LLM)",
          py::arg("video_path"),
          py::arg("interval_seconds") = 3.0,
          py::arg("encode_quality") = 85);
}
