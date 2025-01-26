#pragma once

#include <imgui.h>

#include <EverydayTools/Math/Math.hpp>
#include <ass/enum_map.hpp>
#include <ass/enum_set.hpp>
#include <klgl/events/event_listener_interface.hpp>
#include <klgl/shader/shader.hpp>
#include <klgl/ui/simple_type_widget.hpp>

#include "klgl/application.hpp"
#include "klgl/camera/camera_3d.hpp"
#include "klgl/events/mouse_events.hpp"
#include "klgl/math/rotator.hpp"
#include "klgl/reflection/matrix_reflect.hpp"  // IWYU pragma: keep
#include "klgl/shader/shader.hpp"
#include "klgl/window.hpp"

namespace klgl::gravitoy
{

using namespace edt::lazy_matrix_aliases;  // NOLINT
using Math = edt::Math;

class BodyInfo
{
public:
    Vec3f orbit_center;
    float orbit_radius;
    Rotator initial_rotation;
    Rotator rotation_per_second;
    Rotator rotation;
};

class GravitoyApp : public Application
{
public:
    static std::vector<Vec3f> CalculateInitialParticePositions();

    std::tuple<int, int> GetOpenGLVersion() const override { return {4, 5}; }

    void Initialize() override;
    void SimulationTimeStep();
    void RenderWorld();

    std::span<const edt::Vec3f> UpdateBodiesPositions();

    void Tick() override;

    void OnMouseMove(const events::OnMouseMove& event);

    void RenderGUI();

    void HandleInput();

    static constexpr size_t kTotalParticles = 1'000'000;

    int time_steps_per_frame_ = 30;
    float camera_speed_ = 5.f;
    float time_step_ = 0.f;
    Camera3d camera_{Vec3f{0, 65, 0}, {.yaw = -90, .pitch = 0}};

    GlVertexArrayId particles_vao_;
    GlBufferId particels_positions_buffer_;
    GlBufferId particles_velocities_buffer_;

    GlVertexArrayId bodies_vao_;
    GlBufferId bodies_positions_buffer_;

    std::shared_ptr<Shader> compute_shader_;
    std::shared_ptr<Shader> particle_shader_;
    std::shared_ptr<Shader> body_shader_;

    UniformHandle u_body_a_pos_ = UniformHandle("BlackHolePos1");
    UniformHandle u_body_b_pos_ = UniformHandle("BlackHolePos2");
    UniformHandle u_delta_t_ = UniformHandle("u_delta_t");

    UniformHandle u_particle_color_ = UniformHandle("u_color");
    UniformHandle u_particle_mvp_ = UniformHandle("u_mvp");

    UniformHandle u_body_color_ = UniformHandle("u_color");
    UniformHandle u_body_mvp_ = UniformHandle("u_mvp");

    std::vector<Vec3f> bodies_positions_;

    float particle_alpha_ = 0.1f;

    std::unique_ptr<events::IEventListener> event_listener_;
    std::array<BodyInfo, 2> bodies_{
        BodyInfo{
            .orbit_center{0, 0, 0},
            .orbit_radius = 5,
            .initial_rotation{.pitch = 0},
            .rotation_per_second{.yaw = 500, .pitch = 600, .roll = 700},
            .rotation{},
        },
        BodyInfo{
            .orbit_center{0, 0, 0},
            .orbit_radius = 5,
            .initial_rotation{.pitch = 180},
            .rotation_per_second{.yaw = 500, .pitch = 600, .roll = 700},
            .rotation{},
        },
    };
};

}  // namespace klgl::gravitoy
