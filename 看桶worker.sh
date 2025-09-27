# 0102已经验证这里的四个桶都有权限
rlaunch \
    --positive-tags feature/gpfs=yes \
    --cpu 30 \
    --memory $((200*1024))  \
    --mount=juicefs+s3://oss.i.shaipower.com/yuxiao/:/mnt/yuxiao-juicefs/ \
    --mount=juicefs+s3://oss.i.shaipower.com/dsh-jfs:/mnt/step2_alignment_jfs \
    --mount=juicefs+s3://oss.i.shaipower.com/step2-alignment-jfs:/mnt/step2-alignment-jfs \
    --mount=juicefs+s3://oss.i.shaipower.com/mrh-jfs:/mnt/mrh-jfs \
    --charged-group=step1o \
    --private-machine=group \
    -- zsh
