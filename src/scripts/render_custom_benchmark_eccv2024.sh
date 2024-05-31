export LIGHTING_ITENSITY=1.0 # lighting intensity
export RADIUS=0.4 # distance to camera

MODEL_ROOT="/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/models"
OUTPUT_ROOT="/home/jikaiwang/Datasets/HOT_Dataset/object_detection_benchmark/model_features"
MODEL_IDS=(
G01_1
G01_2
G01_3
G01_4
G02_1
G02_2
G02_3
G02_4
G04_1
G04_2
G04_3
G04_4
G05_1
G05_2
G05_3
G05_4
G06_1
G06_2
G06_3
G06_4
G07_1
G07_2
G07_3
G07_4
G09_1
G09_2
G09_3
G09_4
G10_1
G10_2
G10_3
G10_4
G11_1
G11_2
G11_3
G11_4
G15_1
G15_2
G15_3
G15_4
G16_1
G16_2
G16_3
G16_4
G18_1
G18_2
G18_3
G18_4
G19_1
G19_2
G19_3
G19_4
G20_1
G20_2
G20_3
G20_4
G21_1
G21_2
G21_3
G21_4
G22_1
G22_2
G22_3
G22_4
)

for MODEL_ID in ${MODEL_IDS[@]} ; do
    echo "###############################################################################"
    echo "# Run Rendering on ${MODEL_ID}"
    echo "###############################################################################"
    OUTPUT_DIR=$OUTPUT_ROOT/$MODEL_ID
    mkdir -p $OUTPUT_DIR
    python -m src.poses.pyrender \
        $MODEL_ROOT/$MODEL_ID/"textured_mesh.obj" \
        ./src/poses/predefined_poses/obj_poses_level1.npy \
        $OUTPUT_DIR \
        0 \
        False \
        $LIGHTING_ITENSITY \
        $RADIUS
done