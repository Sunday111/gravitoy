layout(location = 0) in vec4 a_position;

uniform vec4 u_color;
uniform mat4 u_mvp;

out vec4 Color;

void main()
{
    gl_Position = u_mvp * a_position;
    Color = u_color;
}
