#pragma once

#include <imgui.h>

#include <EverydayTools/Math/Math.hpp>
#include <ass/enum_map.hpp>
#include <ass/enum_set.hpp>
#include <klgl/events/event_listener_interface.hpp>
#include <klgl/mesh/mesh_data.hpp>
#include <klgl/shader/shader.hpp>
#include <klgl/ui/simple_type_widget.hpp>

#include "klgl/application.hpp"
#include "klgl/camera/camera_3d.hpp"
#include "klgl/error_handling.hpp"
#include "klgl/events/mouse_events.hpp"
#include "klgl/math/rotator.hpp"
#include "klgl/reflection/matrix_reflect.hpp"  // IWYU pragma: keep
#include "klgl/shader/shader.hpp"
#include "klgl/texture/texture.hpp"
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

class Framebuffer
{
public:
    void Bind(GlFramebufferBindTarget target = GlFramebufferBindTarget::DrawAndRead)
    {
        OpenGl::BindFramebuffer(target, fbo);
    }

    void CreateWithResolution(const edt::Vec2<size_t>& resolution)
    {
        if (fbo.IsValid())
        {
            OpenGl::DeleteFramebuffer(fbo);
            fbo = {};

            depth_stencil = nullptr;

            OpenGl::DeleteRenderbuffer(rbo_depth_stencil);
            rbo_depth_stencil = {};
        }

        rbo_depth_stencil = OpenGl::GenRenderbuffer();
        OpenGl::BindRenderbuffer(rbo_depth_stencil);
        OpenGl::RenderbufferStorage(GlTextureInternalFormat::DEPTH24_STENCIL8, resolution);

        color = Texture::CreateEmpty(resolution, GlTextureInternalFormat::RGB32F);
        color->Bind();
        OpenGl::SetTextureMinFilter(GlTargetTextureType::Texture2d, GlTextureFilter::Nearest);
        OpenGl::SetTextureMagFilter(GlTargetTextureType::Texture2d, GlTextureFilter::Nearest);
        for (const auto axis : ass::EnumSet<GlTextureWrapAxis>::Full())
        {
            OpenGl::SetTextureWrap(GlTargetTextureType::Texture2d, axis, GlTextureWrapMode::ClampToBorder);
        }

        fbo = OpenGl::GenFramebuffer();
        Bind();

        // Color
        OpenGl::FramebufferTexture2D(
            GlFramebufferBindTarget::DrawAndRead,
            GlFramebufferAttachment::Color0,
            GlTargetTextureType::Texture2d,
            color->GetTexture());

        // Depth stencil
        OpenGl::FramebufferRenderbuffer(
            GlFramebufferBindTarget::DrawAndRead,
            GlFramebufferAttachment::DepthStencil,
            rbo_depth_stencil);

        ErrorHandling::Ensure(
            glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,
            "Incomplete frambuffer!");
        OpenGl::BindFramebuffer(GlFramebufferBindTarget::DrawAndRead, {});
    }

    GlRenderbufferId rbo_depth_stencil;
    GlFramebufferId fbo{};
    std::unique_ptr<Texture> color{};
    std::unique_ptr<Texture> depth_stencil{};
};

class GravitoyApp : public Application
{
public:
    static std::vector<Vec4f> CalculateInitialParticePositions();

    std::tuple<int, int> GetOpenGLVersion() const override { return {4, 5}; }

    void Initialize() override;
    void SimulationTimeStep();
    void RenderWorld();

    std::span<const edt::Vec4f> UpdateBodiesPositions();

    void Tick() override;

    void OnMouseMove(const events::OnMouseMove& event);

    void RenderGUI();

    void HandleInput();

    void UpdateFramebuffers();

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

    std::shared_ptr<Shader> textured_quad_shader_;
    UniformHandle u_textured_quad_shader_texture_ = UniformHandle("u_texture");

    std::shared_ptr<Shader> blur_shader_;
    UniformHandle u_blur_shader_texture_ = UniformHandle("u_texture");
    UniformHandle u_blur_shader_horizontal_ = UniformHandle("u_horizontal");

    std::vector<Vec4f> bodies_positions_;

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

    std::shared_ptr<MeshOpenGL> mesh_;
    edt::Vec2<size_t> fbo_resolution_{};
    std::array<Framebuffer, 2> framebuffers_{};
};

}  // namespace klgl::gravitoy
