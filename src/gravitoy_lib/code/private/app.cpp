#include "gravitoy/app.hpp"

#include <klgl/shader/shader_define.hpp>

#include "EverydayTools/Math/SurfacePoints.hpp"
#include "klgl/events/event_listener_method.hpp"
#include "klgl/events/event_manager.hpp"
#include "klgl/mesh/procedural_mesh_generator.hpp"
#include "klgl/opengl/gl_api.hpp"
#include "klgl/opengl/vertex_attribute_helper.hpp"
#include "klgl/template/register_attribute.hpp"

namespace klgl::gravitoy
{

struct MeshVertex
{
    edt::Vec2f position{};
    edt::Vec2f texture_coordinates{};
};

std::vector<float> GenerateGaussianKernel(size_t kernelSize, float sigma)
{
    assert(kernelSize % 2 == 1);  // expected to be odd number
    size_t halfSize = kernelSize / 2;
    std::vector<float> kernel(halfSize + 1);

    float variance = sigma * sigma;
    float denominator = std::sqrt(2 * std::numbers::pi_v<float> * variance);

    for (size_t i = 0; i <= halfSize; i++)
    {
        float x = static_cast<float>(i);
        kernel[i] = std::exp(-x * x / variance) / denominator;
    }

    float sum = kernel.front();
    for (float v : kernel | std::views::drop(1))
    {
        sum += 2 * v;
    }

    for (float& v : kernel)
    {
        v /= sum;
    }

    return kernel;
}

void GravitoyApp::UpdateBlurShader()
{
    blur_weights_ = GenerateGaussianKernel(2 * max_blur_offset_ + 1, blur_sigma_);
    auto d = blur_shader_->GetDefine(Name{"NUM_BLUR_WEIGHTS"});
    blur_shader_->SetDefineValue(d, static_cast<int>(max_blur_offset_ + 1));
    std::string compile_buffer;
    blur_shader_->Compile(compile_buffer);

    blur_shader_->Use();

    std::string uniform_name;
    for (size_t i = 0; i != blur_weights_.size(); ++i)
    {
        uniform_name.clear();
        std::format_to(std::back_inserter(uniform_name), "u_blur_weights[{}]", i);
        auto loc = blur_shader_->GetUniform(Name(uniform_name));
        blur_shader_->SetUniform(loc, blur_weights_[i]);
        blur_shader_->SendUniform(loc);
    }
}

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
    textured_quad_shader_ = std::make_unique<Shader>("gravitoy/textured_quad");
    blur_shader_ = std::make_unique<Shader>("gravitoy/blur");

    UpdateBlurShader();

    {
        auto d = compute_shader_->GetDefine(Name{"NUM_BODIES"});
        const int num_bodies = static_cast<int>(bodies_.size());
        compute_shader_->SetDefineValue(d, num_bodies);
        std::string compile_buffer;
        compute_shader_->Compile(compile_buffer);
    }

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

    {
        // Create quad mesh
        const auto mesh_data = klgl::ProceduralMeshGenerator::GenerateQuadMesh();

        std::vector<MeshVertex> vertices;
        vertices.reserve(mesh_data.vertices.size());
        for (size_t i = 0; i != mesh_data.vertices.size(); ++i)
        {
            vertices.emplace_back(MeshVertex{
                .position = mesh_data.vertices[i],
                .texture_coordinates = mesh_data.texture_coordinates[i],
            });
        }

        mesh_ = klgl::MeshOpenGL::MakeFromData(std::span{vertices}, std::span{mesh_data.indices}, mesh_data.topology);
        mesh_->Bind();

        // Declare vertex buffer layout
        klgl::RegisterAttribute<&MeshVertex::position>(0);
        klgl::RegisterAttribute<&MeshVertex::texture_coordinates>(1);
    }
}

void GravitoyApp::UpdateFramebuffers()
{
    const auto resolution = GetWindow().GetSize().Cast<size_t>();
    if (resolution == fbo_resolution_ && resolution * msaa_ == fbo_upscaled_resolution_)
    {
        return;
    }

    fbo_resolution_ = resolution;
    fbo_upscaled_resolution_ = fbo_resolution_ * msaa_;
    for (auto& f : blur_framebuffers_)
    {
        f.CreateWithResolution(fbo_upscaled_resolution_);
    }
}

std::vector<Vec4f> GravitoyApp::CalculateInitialParticePositions() const
{
    std::vector<Vec4f> positions;
    positions.reserve(kTotalParticles);

    edt::SurfacePointsUtilities::HeartSurface(
        kTotalParticles,
        1.f,
        [&](Vec3f v) { positions.push_back(Vec4f(v, 1.f)); });

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

        glUniform4fv(
            static_cast<GLint>(compute_shader_->GetUniformLocation("Bodies")),
            static_cast<GLsizei>(bodies_positions_.size()),
            bodies_positions_.front().data());
        compute_shader_->SetUniform(u_delta_t_, time_step_);
        compute_shader_->SendUniforms();
        glDispatchCompute(kTotalParticles, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
}

void GravitoyApp::RenderWorld()
{
    UpdateFramebuffers();
    auto render_scene = [&]
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
    };

    // Render scene to texture
    {
        blur_framebuffers_[0].Bind();
        OpenGl::Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        OpenGl::Viewport(
            0,
            0,
            static_cast<GLsizei>(fbo_upscaled_resolution_.x()),
            static_cast<GLsizei>(fbo_upscaled_resolution_.y()));
        render_scene();
    }

    // Pass 1: Do blur passes
    blur_shader_->Use();

    for (size_t i = 0; i != num_blur_passes_ * 2; ++i)
    {
        uint32_t is_horizontal = i % 2 != 0;

        blur_framebuffers_[(i + 1) % 2].Bind();
        OpenGl::Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        auto& texture = *blur_framebuffers_[i % 2].color;

        blur_shader_->SetUniform(u_blur_shader_horizontal_, is_horizontal);
        blur_shader_->SetUniform(u_blur_shader_texture_, texture);
        blur_shader_->SendUniforms();

        texture.Bind();
        mesh_->BindAndDraw();
    }

    // Render scene again over blurred image
    {
        blur_framebuffers_.front().Bind();
        render_scene();
    }

    // Render to screen
    {
        auto& texture = *blur_framebuffers_.front().color;
        OpenGl::BindFramebuffer(GlFramebufferBindTarget::DrawAndRead, {});
        OpenGl::Viewport(0, 0, static_cast<GLsizei>(fbo_resolution_.x()), static_cast<GLsizei>(fbo_resolution_.y()));
        OpenGl::Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        textured_quad_shader_->Use();
        textured_quad_shader_->SetUniform(u_textured_quad_shader_texture_, texture);
        textured_quad_shader_->SendUniforms();
        texture.Bind();
        mesh_->BindAndDraw();
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

        int msaa = static_cast<int>(msaa_);
        if (ImGui::SliderInt("MSAA", &msaa, 1, 4))
        {
            msaa_ = static_cast<size_t>(msaa);
        }

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

        if (ImGui::CollapsingHeader("Blur"))
        {
            if (ImGui::SliderFloat("Intensity", &blur_sigma_, 0.1f, 10.f))
            {
                UpdateBlurShader();
            }

            int blur_offset = static_cast<int>(max_blur_offset_);
            if (ImGui::SliderInt("Area", &blur_offset, 1, 16))
            {
                max_blur_offset_ = static_cast<size_t>(blur_offset);
                UpdateBlurShader();
            }

            int n_passes = static_cast<int>(num_blur_passes_);
            if (ImGui::SliderInt("Passes", &n_passes, 1, 30))
            {
                num_blur_passes_ = static_cast<size_t>(n_passes);
            }

            if (ImGui::CollapsingHeader("Shader"))
            {
                blur_shader_->DrawDetails();
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
