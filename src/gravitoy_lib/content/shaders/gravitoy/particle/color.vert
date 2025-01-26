layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_velocity;

uniform vec4 u_color;
uniform mat4 u_mvp;

out vec4 Color;

#ifndef COLOR_FUNCTION
    #define COLOR_FUNCTION 0
#endif

void main()
{
    gl_Position = u_mvp * vec4(a_position, 1);
    #if COLOR_FUNCTION == 1
        vec3 v = abs(normalize(a_velocity));
        float mc = max(max(v.x, max(v.y, v.z)), 0.1);
        Color = vec4(v / mc, u_color.a);
    #elif COLOR_FUNCTION == 2
        vec4 slow = vec4(1, 0, 0, u_color.a);
        vec4 fast = vec4(1, 1, 1, u_color.a);
        float speed = length(a_velocity);
        Color = mix(slow, fast, clamp(speed / 350.f, 0.f, 1.f));
    #else
        Color = u_color;
        Color.x += min(abs(a_velocity.x), 0);
    #endif
}
