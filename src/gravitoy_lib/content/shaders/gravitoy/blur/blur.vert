layout(location = 0) in vec2 a_vertex;
layout(location = 1) in vec2 a_tex_coord;

uniform vec4 u_color;

out vec4 Color;
out vec2 TexCoord;

void main()
{
    gl_Position = vec4(a_vertex, 0.0, 1.0);
    Color = u_color;
    TexCoord = a_tex_coord;
}
