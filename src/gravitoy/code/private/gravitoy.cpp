#include <imgui.h>

#include <EverydayTools/Math/Math.hpp>
#include <ass/enum_map.hpp>
#include <ass/enum_set.hpp>
#include <klgl/shader/shader.hpp>
#include <klgl/ui/simple_type_widget.hpp>

#include "klgl/application.hpp"
#include "klgl/camera/camera_3d.hpp"
#include "klgl/error_handling.hpp"
#include "klgl/events/event_listener_method.hpp"
#include "klgl/events/event_manager.hpp"
#include "klgl/events/mouse_events.hpp"
#include "klgl/math/rotator.hpp"
#include "klgl/opengl/gl_api.hpp"
#include "klgl/opengl/vertex_attribute_helper.hpp"
#include "klgl/reflection/matrix_reflect.hpp" // IWYU pragma: keep
#include "klgl/shader/shader.hpp"
#include "klgl/window.hpp"

namespace klgl::compute_shader_example {

using namespace edt::lazy_matrix_aliases; // NOLINT
using Math = edt::Math;

class BodyInfo {
public:
  Vec3f orbit_center;
  float orbit_radius;
  Rotator initial_rotation;
  Rotator rotation_per_second;
  Rotator rotation;
};

class ComputeShaderExampleApp : public Application {
  std::tuple<int, int> GetOpenGLVersion() const override { return {4, 5}; }

  void Initialize() override {
    Application::Initialize();
    OpenGl::SetClearColor({});
    GetWindow().SetSize(1000, 1000);
    GetWindow().SetTitle("Painter 2d");
    SetTargetFramerate({60.f});
    event_listener_ = events::EventListenerMethodCallbacks<
        &ComputeShaderExampleApp::OnMouseMove>::CreatePtr(this);
    GetEventManager().AddEventListener(*event_listener_);

    compute_shader_ = std::make_unique<Shader>("gravitoy/compute_shader");
    particle_shader_ = std::make_unique<Shader>("gravitoy/particle");
    body_shader_ = std::make_unique<Shader>("gravitoy/body");

    particles_vao_ = OpenGl::GenVertexArray();
    OpenGl::BindVertexArray(particles_vao_);
    {
      const size_t a_particle_shader_position =
          particle_shader_->GetInfo()
              .VerifyAndGetVertexAttributeLocation<edt::Vec3f>("a_position");
      particels_positions_buffer_ = OpenGl::GenBuffer();
      const auto positions = CalculateInitialParticePositions();
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0,
                       particels_positions_buffer_.GetValue());
      OpenGl::BufferData(GlBufferType::ShaderStorage, std::span{positions},
                         GlUsage::DynamicDraw);
      OpenGl::BindBuffer(GlBufferType::Array, particels_positions_buffer_);
      OpenGl::EnableVertexAttribArray(a_particle_shader_position);
      VertexBufferHelperStatic<edt::Vec4f, false>::AttributePointer(
          a_particle_shader_position);
    }

    {
      const size_t a_particle_shader_velocity =
          particle_shader_->GetInfo()
              .VerifyAndGetVertexAttributeLocation<edt::Vec3f>("a_velocity");
      particles_velocities_buffer_ = OpenGl::GenBuffer();
      std::vector<Vec3f> velocities(kTotalParticles, Vec3f{});
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1,
                       particles_velocities_buffer_.GetValue());
      OpenGl::BufferData(GlBufferType::ShaderStorage, std::span{velocities},
                         GlUsage::DynamicCopy);
      OpenGl::BindBuffer(GlBufferType::Array, particles_velocities_buffer_);
      OpenGl::EnableVertexAttribArray(a_particle_shader_velocity);
      VertexBufferHelperStatic<edt::Vec4f, false>::AttributePointer(
          a_particle_shader_velocity);
    }
    OpenGl::BindVertexArray({});

    {
      for (auto &body : bodies_) {
        body.rotation = body.initial_rotation;
      }

      bodies_positions_buffer_ = OpenGl::GenBuffer();
      OpenGl::BindBuffer(GlBufferType::Array, bodies_positions_buffer_);
      OpenGl::BufferData(GlBufferType::Array, UpdateBodiesPositions(),
                         GlUsage::DynamicDraw);

      bodies_vao_ = OpenGl::GenVertexArray();
      const size_t a_body_shader_position =
          body_shader_->GetInfo()
              .VerifyAndGetVertexAttributeLocation<edt::Vec3f>("a_position");
      OpenGl::BindVertexArray(bodies_vao_);
      OpenGl::BindBuffer(GlBufferType::Array, bodies_positions_buffer_);
      OpenGl::VertexAttribPointer(a_body_shader_position, 3,
                                  GlVertexAttribComponentType::Float, false, 0,
                                  nullptr);
      OpenGl::EnableVertexAttribArray({});

      OpenGl::BindVertexArray({});
    }
  }

  static std::vector<Vec3f> CalculateInitialParticePositions() {
    std::vector<Vec3f> positions(kTotalParticles);

    const size_t s = static_cast<size_t>(
        std::round(std::pow(static_cast<float>(kTotalParticles), 1.f / 3)));
    auto delta = Vec3f{} + 2.f / static_cast<float>(s);

    [&] {
      size_t i = 0;
      for (size_t x = 0; x < s; x++) {
        for (size_t y = 0; y < s; y++) {
          for (size_t z = 0; z < s; z++) {
            positions[i] = (Vec3<size_t>{x, y, z}.Cast<float>() * delta - 1);
            if (++i == kTotalParticles) {
              return;
            }
          }
        }
      }
    }();

    return positions;
  }

  std::span<const edt::Vec3f> UpdateBodiesPositions() {
    bodies_positions_.clear();
    for (BodyInfo &body : bodies_) {
      bodies_positions_.push_back(Math::TransformPos(
          body.rotation.ToMatrix(), Vec3f{body.orbit_radius, 0, 0}));
    }

    return bodies_positions_;
  }

  void Tick() override {
    HandleInput();

    for ([[maybe_unused]] const int i :
         std::views::iota(0, time_steps_per_frame_)) {
      for (BodyInfo &body : bodies_) {
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

    OpenGl::EnableBlending();
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const auto mvp = camera_.GetViewMatrix().MatMul(
        camera_.GetProjectionMatrix(GetWindow().GetAspect()));

    {
      // Draw the particles
      particle_shader_->Use();
      particle_shader_->SetUniform(u_particle_mvp_, mvp);
      particle_shader_->SetUniform(u_particle_color_,
                                   Vec4f{1, 1, 1, particle_alpha_});
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
      OpenGl::BufferSubData(GlBufferType::Array, 0,
                            std::span{bodies_positions_});

      OpenGl::PointSize(15.f);
      OpenGl::BindVertexArray(bodies_vao_);
      OpenGl::DrawArrays(GlPrimitiveType::Points, 0, bodies_positions_.size());
    }

    RenderGUI();
  }

  void OnMouseMove(const events::OnMouseMove &event) {
    constexpr float sensitivity = 0.01f;
    if (GetWindow().IsFocused() && GetWindow().IsInInputMode() &&
        !ImGui::GetIO().WantCaptureMouse) {
      const auto delta = (event.current - event.previous) * sensitivity;
      const auto [yaw, pitch, roll] = camera_.GetRotation();
      camera_.SetRotation({yaw + delta.x(), pitch + delta.y(), roll});
    }
  }

  void RenderGUI() {
    if (ImGui::Begin("Settings")) {
      camera_.Widget();
      ImGui::Separator();
      SimpleTypeWidget("Camera speed", camera_speed_);
      const auto framerate = static_cast<size_t>(GetFramerate());
      SimpleTypeWidget("Framerate", framerate);
      ImGui::SliderFloat("Time step", &time_step_, 0.0f, 1.f / 10000, "%.6f");
      ImGui::SliderInt("Time steps per frame", &time_steps_per_frame_, 0, 40);
      ImGui::SliderFloat("Particle alpha", &particle_alpha_, 0.0001f, 1.f,
                         "%.4f");

      if (ImGui::CollapsingHeader("Bodies")) {
        auto rotator_widget = [](std::string_view title, Rotator &rotator) {
          if (ImGui::CollapsingHeader(title.data())) {
            SimpleTypeWidget("yaw", rotator.yaw);
            SimpleTypeWidget("pitch", rotator.pitch);
            SimpleTypeWidget("roll", rotator.roll);
          }
        };

        for (BodyInfo &body : bodies_) {
          ImGui::PushID(&body);
          if (ImGui::CollapsingHeader("Body")) {
            SimpleTypeWidget("Orbit center", body.orbit_center);
            SimpleTypeWidget("Orbit radius", body.orbit_radius);
            rotator_widget("Initial rotation", body.initial_rotation);
            rotator_widget("Rotation per second", body.rotation_per_second);
            rotator_widget("Current rotation", body.rotation);
          }
          ImGui::PopID();
        }
      }

      if (ImGui::CollapsingHeader("Shader")) {
        particle_shader_->DrawDetails();
      }
    }

    ImGui::End();
  }

  void HandleInput() {
    if (!ImGui::GetIO().WantCaptureKeyboard) {
      int right = 0;
      int forward = 0;
      int up = 0;
      if (ImGui::IsKeyDown(ImGuiKey_W))
        forward += 1;
      if (ImGui::IsKeyDown(ImGuiKey_S))
        forward -= 1;
      if (ImGui::IsKeyDown(ImGuiKey_D))
        right += 1;
      if (ImGui::IsKeyDown(ImGuiKey_A))
        right -= 1;
      if (ImGui::IsKeyDown(ImGuiKey_E))
        up += 1;
      if (ImGui::IsKeyDown(ImGuiKey_Q))
        up -= 1;
      if (std::abs(right) + std::abs(forward) + std::abs(up)) {
        Vec3f delta = static_cast<float>(forward) * camera_.GetForwardAxis();
        delta += static_cast<float>(right) * camera_.GetRightAxis();
        delta += static_cast<float>(up) * camera_.GetUpAxis();
        camera_.SetEye(camera_.GetEye() +
                       delta * camera_speed_ * GetLastFrameDurationSeconds());
      }
    }
  }

  static constexpr size_t kTotalParticles = 1'000'000;

  int time_steps_per_frame_ = 30;
  float camera_speed_ = 5.f;
  float time_step_ = 0.f;
  Camera3d camera_{Vec3f{0, 15, 0}, {.yaw = -90, .pitch = 0}};

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

void Main() {
  ComputeShaderExampleApp app;
  app.Run();
}
} // namespace klgl::compute_shader_example

int main() {
  klgl::ErrorHandling::InvokeAndCatchAll(klgl::compute_shader_example::Main);
  return 0;
}
