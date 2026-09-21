#version 330

uniform sampler2D base_color;
uniform int mode;
uniform bool wire;
in vec3 rgb;
flat in vec3 face_normal;
in vec2 uv;
out vec4 color;

void main() {
    vec3 result = rgb * texture(base_color, uv).rgb;
    if (mode == 1) result = face_normal * 0.5 + 0.5;
    if (wire) result = vec3(0.15, 0.55, 0.95);
    color = vec4(result, 1);
}
