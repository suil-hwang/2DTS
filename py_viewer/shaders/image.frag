#version 330

uniform sampler2D image;
in vec2 uv;
out vec4 color;

void main() {
    color = vec4(texture(image, uv).rgb, 1);
}
