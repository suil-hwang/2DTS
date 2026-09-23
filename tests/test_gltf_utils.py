import unittest

import numpy as np

from src.diff_recon.utils.gltf_utils import read_accessor


class ReadAccessorTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        vertices = np.empty(5, dtype=[("position", "<f4", 3), ("uv", "<f4", 2)])  # interleaved, 20-byte stride
        vertices["position"] = rng.normal(size=(5, 3))
        vertices["uv"] = rng.random((5, 2))
        colors = np.zeros(4, dtype=[("rgb", "u1", 3), ("pad", "u1")])  # 4-byte stride for 3 components
        colors["rgb"] = rng.integers(0, 256, size=(4, 3))
        self.vertices, self.colors = vertices, colors

        self.bin_chunk = b"\xff" * 8 + vertices.tobytes() + colors.tobytes()[:-1]  # the last element ends the buffer
        self.gltf = {
            "bufferViews": [{"byteOffset": 8, "byteStride": 20}, {"byteOffset": 8 + vertices.nbytes, "byteStride": 4}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "type": "VEC3", "count": 5},
                {"bufferView": 0, "byteOffset": 12, "componentType": 5126, "type": "VEC2", "count": 5},
                {"bufferView": 1, "componentType": 5121, "type": "VEC3", "count": 4, "normalized": True},
                {"bufferView": 1, "componentType": 5121, "type": "VEC3", "count": 0},
            ],
        }

    def test_reads_interleaved_attributes(self):
        positions = read_accessor(self.gltf, self.bin_chunk, 0)
        uvs = read_accessor(self.gltf, self.bin_chunk, 1)

        np.testing.assert_array_equal(positions, self.vertices["position"])
        np.testing.assert_array_equal(uvs, self.vertices["uv"])
        self.assertEqual(positions.dtype, np.float32)
        self.assertTrue(positions.flags.writeable)

    def test_reads_padded_normalized_components_up_to_the_end_of_the_buffer(self):
        colors = read_accessor(self.gltf, self.bin_chunk, 2)
        np.testing.assert_array_equal(colors, self.colors["rgb"].astype(np.float32) / 255.0)

    def test_reads_an_empty_strided_accessor(self):
        self.assertEqual(read_accessor(self.gltf, self.bin_chunk, 3).shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
