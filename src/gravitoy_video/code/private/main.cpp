#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "EverydayTools/Math/SurfacePoints.hpp"
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

class GravitoyVideoApp : public GravitoyApp
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
                .initial_rotation{.yaw = -90.f},
                .rotation_per_second = {},
                .rotation{},
            },
            BodyInfo{
                .orbit_center{0, 0, 0},
                .orbit_radius = 5,
                .initial_rotation{.yaw = 90.f},
                .rotation_per_second = {},
                .rotation{},
            },
            BodyInfo{
                .orbit_center{0, 0, 0},
                .orbit_radius = 5,
                .initial_rotation{.pitch = 0},
                .rotation_per_second = {},
                .rotation{},
            },
            BodyInfo{
                .orbit_center{0, 0, 0},
                .orbit_radius = 5,
                .initial_rotation{.pitch = 180},
                .rotation_per_second = {},
                .rotation{},
            },
            BodyInfo{
                .orbit_center{0, 0, 0},
                .orbit_radius = 5,
                .initial_rotation{.yaw = 90, .pitch = -90},
                .rotation_per_second = {},
                .rotation{},
            },
            BodyInfo{
                .orbit_center{0, 0, 0},
                .orbit_radius = 5,
                .initial_rotation{.yaw = 90, .pitch = 90},
                .rotation_per_second = {},
                .rotation{},
            },
        };

        Super::Initialize();
        GetWindow().SetSize(3840, 2160);

        {
            DefineHandle color_def(Name("COLOR_FUNCTION"));
            particle_shader_->SetDefineValue(color_def, 1);
            std::string buf;
            particle_shader_->Compile(buf);
        }

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
            std::lerp(55.f, 55.f, std::min(video_seconds / camera_zoom_out_time_seconds, 1.f));
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

    static Mat4f ComputeRotMtx(Vec3f normal)
    {
        constexpr Vec3f up{0, 0, 1};
        Vec3f axis = up.Cross(normal);  // Rotation axis
        float cos_theta = up.Dot(normal);
        float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);

        auto R = Mat4f::Identity();

        if (sin_theta < 1e-6)
        {
            // If already aligned, use identity matrix
            R(0, 0) = 1;
            R(0, 1) = 0;
            R(0, 2) = 0;
            R(1, 0) = 0;
            R(1, 1) = 1;
            R(1, 2) = 0;
            R(2, 0) = 0;
            R(2, 1) = 0;
            R(2, 2) = 1;
            return R;
        }

        axis.Normalize();  // Ensure the rotation axis is unit length

        // Rodrigues' rotation formula
        float K[3][3] = {{0, -axis.z(), axis.y()}, {axis.z(), 0, -axis.x()}, {-axis.y(), axis.x(), 0}};  // NOLINT

        float I[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};  // NOLINT

        // Compute R = I + sin(theta) * K + (1 - cos(theta)) * K^2
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                R(i, j) = I[i][j] + sin_theta * K[i][j] +
                          (1 - cos_theta) * (K[i][0] * K[0][j] + K[i][1] * K[1][j] + K[i][2] * K[2][j]);
            }
        }

        return R;
    }

    std::vector<Vec4f> CalculateInitialParticePositions() const override
    {
        constexpr size_t kNumHearts = 100;
        static_assert((kTotalParticles % kNumHearts) == 0);
        std::vector<Vec4f> positions;
        positions.reserve(kTotalParticles);

        edt::SurfacePointsUtilities::UniformSphereSurface(
            kNumHearts,
            15.f,
            [&](Vec3f heart_center)
            {
                auto r = ComputeRotMtx(heart_center.Normalized());

                edt::SurfacePointsUtilities::HeartSurface(
                    kTotalParticles / kNumHearts,
                    0.1f,
                    [&](Vec3f p)
                    {
                        p = edt::Math::TransformVector(r, p);
                        positions.push_back(Vec4f(p + heart_center, 1.f));
                    });
            });

        return positions;
    }

    // std::span<const edt::Vec4f> UpdateBodiesPositions() override
    // {
    //     bodies_positions_.clear();
    //     bodies_positions_.push_back({});
    //     // float period = std::numbers::pi_v<float> * 2;
    //     // float scale = 0.33f;

    //     // static float t = 0;
    //     // t += time_step_;

    //     // float dk = period / static_cast<float>(bodies_.size());
    //     // for (size_t i = 0; i != bodies_.size(); ++i)
    //     // {
    //     //     Vec4f p{};
    //     //     float lt = std::fmod(t + dk * static_cast<float>(i), period);
    //     //     p.x() = 16.f * std::pow(std::sin(lt), 3.f);
    //     //     p.z() = 13.f * std::cos(lt) - 5 * std::cos(2 * lt) - 2 * std::cos(3 * lt) - std::cos(4 * lt);
    //     //     p.w() = 1.f;

    //     //     bodies_positions_.push_back(p * scale);
    //     // }

    //     return bodies_positions_;
    // }

    bool WantsToClose() const override { return Super::WantsToClose() || frame_index_ == 60 * 120; }
};

void Main()
{
    GravitoyVideoApp app;
    app.Run();
}
}  // namespace klgl::gravitoy

int main()
{
    klgl::ErrorHandling::InvokeAndCatchAll(klgl::gravitoy::Main);
    return 0;
}
