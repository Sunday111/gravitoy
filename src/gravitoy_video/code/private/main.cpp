#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "fmt/core.h"
#include "fmt/std.h"  // IWYU pragma: keep
#include "gravitoy/app.hpp"
#include "klgl/error_handling.hpp"

namespace klgl::gravitoy
{

static constexpr int FourCC(std::optional<std::string_view> maybe_name)
{
    if (maybe_name.has_value())
    {
        auto name = *maybe_name;
        return cv::VideoWriter::fourcc(name[0], name[1], name[2], name[3]);
    }

    return 0;
}

static constexpr int kOutputVideoFPS = 60;
static constexpr std::string_view kOutputVideoFormat = ".mp4";
static constexpr std::optional<std::string_view> kVideoEncoding = "avc1";

class GravitoryVideoApp : public GravitoyApp
{
public:
    using Super = GravitoyApp;

    std::vector<edt::Vec3u8> frame_pixels_;
    std::unique_ptr<cv::VideoWriter> video_writer_;
    size_t frame_index_ = 0;
    cv::Mat frame_image_;
    float time_scale_ = 1.f / 30.f;
    float camera_rotation_degrees_per_second_ = 30.f;
    float camera_zoom_out_time_seconds = 30.f;

    void Initialize() override
    {
        time_steps_per_frame_ = 100;
        time_step_ = time_scale_ / static_cast<float>(kOutputVideoFPS * time_steps_per_frame_);
        bodies_ = {
            BodyInfo{
                .orbit_center{0, 0, 0},
                .orbit_radius = 5,
                .initial_rotation{.pitch = 0},
                .rotation_per_second = Rotator{.yaw = 25, .pitch = 26, .roll = 27} / time_scale_,
                .rotation{},
            },
            BodyInfo{
                .orbit_center{0, 0, 0},
                .orbit_radius = 5,
                .initial_rotation{.pitch = 180},
                .rotation_per_second = Rotator{.yaw = 25, .pitch = 26, .roll = 27} / time_scale_,
                .rotation{},
            },
        };

        Super::Initialize();
        GetWindow().SetSize(3840, 2160);

        const auto window_size = GetWindow().GetSize();
        const auto window_size_i = window_size.Cast<int>();
        frame_image_ = cv::Mat(window_size_i.y(), window_size_i.x(), CV_8UC3);

        frame_pixels_.resize(window_size.x() * window_size.y());
        const auto output_video_path = (GetExecutableDir() / fmt::format("result{}", kOutputVideoFormat));
        fmt::println("Output file: {}", output_video_path);
        video_writer_ = std::make_unique<cv::VideoWriter>(
            output_video_path.string(),
            FourCC(kVideoEncoding),
            kOutputVideoFPS,
            cv::Size(window_size_i.x(), window_size_i.y()));
    }

    void Tick() override
    {
        AnimateCamera();
        SimulationTimeStep();
        RenderWorld();
        WriteFrame();
        ++frame_index_;
    }

    void AnimateCamera()
    {
        const float video_seconds = static_cast<float>(frame_index_) / static_cast<float>(kOutputVideoFPS);
        const float camera_distance =
            std::lerp(15.f, 85.f, std::min(video_seconds / camera_zoom_out_time_seconds, 1.f));
        camera_.SetEye(Math::TransformPos(
            Rotator{.yaw = video_seconds * camera_rotation_degrees_per_second_}.ToMatrix(),
            Vec3f{0, camera_distance, 0}));
        camera_.SetRotation(Rotator{.yaw = video_seconds * camera_rotation_degrees_per_second_ - 90});
    }

    void WriteFrame()
    {
        const auto window_size = GetWindow().GetSize();

        klgl::OpenGl::ReadPixels(
            0,
            0,
            window_size.x(),
            window_size.y(),
            GL_BGR,
            GL_UNSIGNED_BYTE,
            frame_pixels_.data());

        // Copy the pixel data to a cv::Mat object (OpenCV)
        std::memcpy(frame_image_.data, frame_pixels_.data(), std::span{frame_pixels_}.size_bytes());

        // OpenGL's origin is at the bottom-left corner, so we need to flip the image vertically
        cv::flip(frame_image_, frame_image_, 0);

        video_writer_->write(frame_image_);
    }

    bool WantsToClose() const override { return Super::WantsToClose() || frame_index_ == 60 * 120; }
};

void Main()
{
    GravitoryVideoApp app;
    app.Run();
}
}  // namespace klgl::gravitoy

int main()
{
    klgl::ErrorHandling::InvokeAndCatchAll(klgl::gravitoy::Main);
    return 0;
}
