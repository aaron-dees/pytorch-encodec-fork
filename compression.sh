INPUT_FILE=$1
MODEL_NAME=$2
CHECKPOINT=$3
echo "INPUT WAV FILE -> $INPUT_FILE"
before_dot=${INPUT_FILE%%.*}

if [ -z "$MODEL_NAME"]; then
    MODEL_NAME=encodec_24khz
else
    MODEL_NAME=$MODEL_NAME
fi

if [ "$MODEL_NAME" = "my_encodec" ]; then
    if [ -z "$CHECKPOINT"]; then
        CHECKPOINT=/mnt/lustre/sjtu/home/zkn02/EnCodec_Trainer/outputs/2023-04-11/09-10-26/save/batch4_cut180000_lr5e-05_epoch36_lr5e-05.pt
    else
        CHECKPOINT=$CHECKPOINT
    fi
fi

echo "MODEL NAME -> $MODEL_NAME"

OUTPUT_FILE=./encodec_results/$MODEL_NAME/${before_dot##*/}/
echo "OUTPUT FILE -> $OUTPUT_FILE"

if [ ! -d "$OUTPUT_FILE" ]; then
    # 如果路径不存在，则创建路径
    mkdir -p "$OUTPUT_FILE"
    echo "Created path: $OUTPUT_FILE"
else
    echo "Path already exists: $OUTPUT_FILE"
fi

cp $INPUT_FILE $OUTPUT_FILE

TARGET_BANDWIDTH=(1.5 3 6 12 24)
for bandwidth in ${TARGET_BANDWIDTH[@]}
do 
    OUTPUT_WAV_FILE=$OUTPUT_FILE$bandwidth.wav
    echo "OUTPUT WAV FILE -> $OUTPUT_WAV_FILE"
    python main.py -r -b $bandwidth -f $INPUT_FILE $OUTPUT_WAV_FILE -m $MODEL_NAME -c $CHECKPOINT
done
# encodec [-r] [-b TARGET_BANDWIDTH] [-f] [--hq] [--lm] $INPUT_FILE $OUTPUT_WAV_FILE