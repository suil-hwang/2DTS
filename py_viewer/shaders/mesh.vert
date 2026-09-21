#version 330

uniform mat4 mvp;
in vec3 position;
in vec3 vertex_color;
in vec3 normal;
in vec2 texcoord;
out vec3 rgb;
flat out vec3 face_normal;
out vec2 uv;

void main() {
    gl_Position = mvp * vec4(position, 1);
    rgb = vertex_color;
    face_normal = normal;
    uv = texcoord;
}
