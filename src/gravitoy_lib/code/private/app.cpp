#include "gravitoy/app.hpp"

#include <numbers>

#include "klgl/events/event_listener_method.hpp"
#include "klgl/events/event_manager.hpp"
#include "klgl/opengl/gl_api.hpp"
#include "klgl/opengl/vertex_attribute_helper.hpp"

namespace klgl::gravitoy
{

void GravitoyApp::Initialize()
{
    Application::Initialize();
    OpenGl::SetClearColor({});
    GetWindow().SetSize(1000, 1000);
    GetWindow().SetTitle("Painter 2d");
    SetTargetFramerate({60.f});
    event_listener_ = events::EventListenerMethodCallbacks<&GravitoyApp::OnMouseMove>::CreatePtr(this);
    GetEventManager().AddEventListener(*event_listener_);

    camera_.SetFar(300.f);

    compute_shader_ = std::make_unique<Shader>("gravitoy/compute_shader");
    particle_shader_ = std::make_unique<Shader>("gravitoy/particle");
    body_shader_ = std::make_unique<Shader>("gravitoy/body");

    particles_vao_ = OpenGl::GenVertexArray();
    OpenGl::BindVertexArray(particles_vao_);
    {
        const size_t a_particle_shader_position =
            particle_shader_->GetInfo().VerifyAndGetVertexAttributeLocation<edt::Vec4f>("a_position");
        particels_positions_buffer_ = OpenGl::GenBuffer();
        const auto particles = CalculateInitialParticePositions();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particels_positions_buffer_.GetValue());
        OpenGl::BufferData(GlBufferType::ShaderStorage, std::span{particles}, GlUsage::DynamicDraw);
        OpenGl::BindBuffer(GlBufferType::Array, particels_positions_buffer_);
        OpenGl::EnableVertexAttribArray(a_particle_shader_position);
        VertexBufferHelperStatic<edt::Vec4f, false>::AttributePointer(a_particle_shader_position);
    }

    {
        const size_t a_particle_shader_velocity =
            particle_shader_->GetInfo().VerifyAndGetVertexAttributeLocation<edt::Vec4f>("a_velocity");
        particles_velocities_buffer_ = OpenGl::GenBuffer();
        std::vector<Vec4f> velocities(kTotalParticles, Vec4f{});
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, particles_velocities_buffer_.GetValue());
        OpenGl::BufferData(GlBufferType::ShaderStorage, std::span{velocities}, GlUsage::DynamicCopy);
        OpenGl::BindBuffer(GlBufferType::Array, particles_velocities_buffer_);
        OpenGl::EnableVertexAttribArray(a_particle_shader_velocity);
        VertexBufferHelperStatic<edt::Vec4f, false>::AttributePointer(a_particle_shader_velocity);
    }
    OpenGl::BindVertexArray({});

    {
        for (auto& body : bodies_)
        {
            body.rotation = body.initial_rotation;
        }

        bodies_positions_buffer_ = OpenGl::GenBuffer();
        OpenGl::BindBuffer(GlBufferType::Array, bodies_positions_buffer_);
        OpenGl::BufferData(GlBufferType::Array, UpdateBodiesPositions(), GlUsage::DynamicDraw);

        bodies_vao_ = OpenGl::GenVertexArray();
        const size_t a_body_shader_position =
            body_shader_->GetInfo().VerifyAndGetVertexAttributeLocation<edt::Vec4f>("a_position");
        OpenGl::BindVertexArray(bodies_vao_);
        OpenGl::BindBuffer(GlBufferType::Array, bodies_positions_buffer_);
        OpenGl::VertexAttribPointer(a_body_shader_position, 4, GlVertexAttribComponentType::Float, false, 0, nullptr);
        OpenGl::EnableVertexAttribArray({});

        OpenGl::BindVertexArray({});
    }
}

std::vector<Vec4f> GravitoyApp::CalculateInitialParticePositions()
{
    std::vector<Vec4f> positions;
    positions.reserve(kTotalParticles);

    constexpr float radius = 15.f;
    const double k = std::numbers::pi * 4.0 / (std::sqrt(5.0) + 1);
    const double dn = static_cast<double>(kTotalParticles);
    for (size_t i = 0; i != kTotalParticles; ++i)
    {
        double di = static_cast<double>(i);
        double theta = di * k;
        double cos_phi = 1 - 2 * di / dn;
        double sin_phi = std::sqrt(1 - Math::Sqr(cos_phi));
        double x = std::cos(theta) * sin_phi;
        double y = std::sin(theta) * sin_phi;
        positions.push_back(Vec4f(Vec3<double>{x, y, cos_phi}.Cast<float>() * radius, 1));
    }

    return positions;
}

std::span<const edt::Vec4f> GravitoyApp::UpdateBodiesPositions()
{
    bodies_positions_.clear();
    for (BodyInfo& body : bodies_)
    {
        bodies_positions_.push_back(
            Vec4f(Math::TransformPos(body.rotation.ToMatrix(), Vec3f{body.orbit_radius, 0, 0}), 1));
    }

    return bodies_positions_;
}

void GravitoyApp::Tick()
{
    HandleInput();
    SimulationTimeStep();
    RenderWorld();
    RenderGUI();
}

void GravitoyApp::SimulationTimeStep()
{
    for ([[maybe_unused]] const int i : std::views::iota(0, time_steps_per_frame_))
    {
        for (BodyInfo& body : bodies_)
        {
            body.rotation += body.rotation_per_second * time_step_;
            body.rotation.yaw = std::fmod(body.rotation.yaw, 360.f);
            body.rotation.pitch = std::fmod(body.rotation.pitch, 360.f);
            body.rotation.roll = std::fmod(body.rotation.roll, 360.f);
        }

        UpdateBodiesPositions();

        // Compute particles
        compute_shader_->Use();
        compute_shader_->SetUniform(u_body_a_pos_, bodies_positions_[0]);
        compute_shader_->SetUniform(u_body_b_pos_, bodies_positions_[1]);
        compute_shader_->SetUniform(u_delta_t_, time_step_);
        compute_shader_->SendUniforms();
        glDispatchCompute(kTotalParticles, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
}

void GravitoyApp::RenderWorld()
{
    OpenGl::EnableBlending();
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const auto mvp = camera_.GetViewMatrix().MatMul(camera_.GetProjectionMatrix(GetWindow().GetAspect()));

    {
        // Draw the particles
        particle_shader_->Use();
        particle_shader_->SetUniform(u_particle_mvp_, mvp);
        particle_shader_->SetUniform(u_particle_color_, Vec4f{1, 1, 1, particle_alpha_});
        particle_shader_->SendUniforms();

        OpenGl::PointSize(2.f);
        OpenGl::BindVertexArray(particles_vao_);
        OpenGl::DrawArrays(GlPrimitiveType::Points, 0, kTotalParticles);
    }

    {
        // Draw bodies
        body_shader_->Use();
        body_shader_->SetUniform(u_body_mvp_, mvp);
        body_shader_->SetUniform(u_body_color_, Vec4f{1, 0, 0, 1});
        body_shader_->SendUniforms();

        // Update bodies positions
        OpenGl::BindBuffer(GlBufferType::Array, bodies_positions_buffer_);
        OpenGl::BufferSubData(GlBufferType::Array, 0, std::span{bodies_positions_});

        OpenGl::PointSize(15.f);
        OpenGl::BindVertexArray(bodies_vao_);
        OpenGl::DrawArrays(GlPrimitiveType::Points, 0, bodies_positions_.size());
    }
}

void GravitoyApp::OnMouseMove(const events::OnMouseMove& event)
{
    constexpr float sensitivity = 0.01f;
    if (GetWindow().IsFocused() && GetWindow().IsInInputMode() && !ImGui::GetIO().WantCaptureMouse)
    {
        const auto delta = (event.current - event.previous) * sensitivity;
        const auto [yaw, pitch, roll] = camera_.GetRotation();
        camera_.SetRotation({yaw + delta.x(), pitch + delta.y(), roll});
    }
}

void GravitoyApp::RenderGUI()
{
    if (ImGui::Begin("Settings"))
    {
        camera_.Widget();
        ImGui::Separator();
        SimpleTypeWidget("Camera speed", camera_speed_);
        const auto framerate = static_cast<size_t>(GetFramerate());
        SimpleTypeWidget("Framerate", framerate);
        ImGui::SliderFloat("Time step", &time_step_, 0.0f, 1.f / 10000, "%.6f");
        ImGui::SliderInt("Time steps per frame", &time_steps_per_frame_, 0, 40);
        ImGui::SliderFloat("Particle alpha", &particle_alpha_, 0.0001f, 1.f, "%.4f");

        if (ImGui::CollapsingHeader("Bodies"))
        {
            auto rotator_widget = [](std::string_view title, Rotator& rotator)
            {
                if (ImGui::CollapsingHeader(title.data()))
                {
                    SimpleTypeWidget("yaw", rotator.yaw);
                    SimpleTypeWidget("pitch", rotator.pitch);
                    SimpleTypeWidget("roll", rotator.roll);
                }
            };

            for (BodyInfo& body : bodies_)
            {
                ImGui::PushID(&body);
                if (ImGui::CollapsingHeader("Body"))
                {
                    SimpleTypeWidget("Orbit center", body.orbit_center);
                    SimpleTypeWidget("Orbit radius", body.orbit_radius);
                    rotator_widget("Initial rotation", body.initial_rotation);
                    rotator_widget("Rotation per second", body.rotation_per_second);
                    rotator_widget("Current rotation", body.rotation);
                }
                ImGui::PopID();
            }
        }

        if (ImGui::CollapsingHeader("Shader"))
        {
            particle_shader_->DrawDetails();
        }
    }

    ImGui::End();
}

void GravitoyApp::HandleInput()
{
    if (!ImGui::GetIO().WantCaptureKeyboard)
    {
        int right = 0;
        int forward = 0;
        int up = 0;
        if (ImGui::IsKeyDown(ImGuiKey_W)) forward += 1;
        if (ImGui::IsKeyDown(ImGuiKey_S)) forward -= 1;
        if (ImGui::IsKeyDown(ImGuiKey_D)) right += 1;
        if (ImGui::IsKeyDown(ImGuiKey_A)) right -= 1;
        if (ImGui::IsKeyDown(ImGuiKey_E)) up += 1;
        if (ImGui::IsKeyDown(ImGuiKey_Q)) up -= 1;
        if (std::abs(right) + std::abs(forward) + std::abs(up))
        {
            Vec3f delta = static_cast<float>(forward) * camera_.GetForwardAxis();
            delta += static_cast<float>(right) * camera_.GetRightAxis();
            delta += static_cast<float>(up) * camera_.GetUpAxis();
            camera_.SetEye(camera_.GetEye() + delta * camera_speed_ * GetLastFrameDurationSeconds());
        }
    }
}
}  // namespace klgl::gravitoy
