in vec4 Color;
in vec2 TexCoord;
out vec4 FragColor;

#ifndef NUM_BLUR_WEIGHTS
#define NUM_BLUR_WEIGHTS 5
#endif

uniform sampler2D u_texture;
uniform uint u_horizontal;
uniform float u_blur_weights[NUM_BLUR_WEIGHTS];

void main()
{
    vec2 tex_offset = 1.0 / textureSize(u_texture, 0); // gets size of single texel
    vec3 result = texture(u_texture, TexCoord).rgb * u_blur_weights[0]; // current fragment's contribution

    if(u_horizontal != 0u)
    {
        for(int i = 1; i < NUM_BLUR_WEIGHTS; ++i)
        {
            result += texture(u_texture, TexCoord + vec2(tex_offset.x * i, 0.0)).rgb * u_blur_weights[i];
            result += texture(u_texture, TexCoord - vec2(tex_offset.x * i, 0.0)).rgb * u_blur_weights[i];
        }
    }
    else
    {
        for(int i = 1; i < NUM_BLUR_WEIGHTS; ++i)
        {
            result += texture(u_texture, TexCoord + vec2(0.0, tex_offset.y * i)).rgb * u_blur_weights[i];
            result += texture(u_texture, TexCoord - vec2(0.0, tex_offset.y * i)).rgb * u_blur_weights[i];
        }
    }

    FragColor = vec4(result, 1.0);
}
