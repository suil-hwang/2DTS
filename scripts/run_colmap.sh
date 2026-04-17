#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Project path is required"
    echo "Usage: $0 <project_path>"
    exit 1
fi

PROJECT_PATH=$1

# Create necessary directories if they don't exist
mkdir -p $PROJECT_PATH/sparse/1

colmap feature_extractor \
    --database_path $PROJECT_PATH/database.db \
    --image_path $PROJECT_PATH/images \
    --SiftExtraction.estimate_affine_shape true \
    --SiftExtraction.domain_size_pooling true \

# colmap image_registrator \
#     --database_path $PROJECT_PATH/database.db \
#     --input_path $PROJECT_PATH/sparse/0 \
#     --output_path $PROJECT_PATH/sparse/0

colmap exhaustive_matcher \
    --database_path $PROJECT_PATH/database.db \
    --ExhaustiveMatching.block_size 100 \
    --SiftMatching.guided_matching true

colmap point_triangulator \
    --database_path $PROJECT_PATH/database.db \
    --image_path $PROJECT_PATH/images \
    --input_path $PROJECT_PATH/sparse/0 \
    --output_path $PROJECT_PATH/sparse/1

colmap model_converter \
    --input_path $PROJECT_PATH/sparse/1 \
    --output_path $PROJECT_PATH/sparse/1 \
    --output_type TXT

colmap image_undistorter \
    --image_path $PROJECT_PATH/images \
    --input_path $PROJECT_PATH/sparse/1 \
    --output_path $PROJECT_PATH/dense

colmap patch_match_stereo \
    --workspace_path $PROJECT_PATH/dense

colmap stereo_fusion \
    --workspace_path $PROJECT_PATH/dense \
    --output_path $PROJECT_PATH/dense/fused.ply