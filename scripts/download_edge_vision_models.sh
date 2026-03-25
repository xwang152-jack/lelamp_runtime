#!/bin/bash
#
# LeLamp 边缘视觉模型下载脚本
# 下载 EfficientDet-Lite0 物体检测模型
#
# 使用方法:
#   ./download_edge_vision_models.sh [模型目录]
#
# 默认模型目录: ~/lelamp_runtime/models
#

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
MODEL_NAME="efficientdet_lite0.tflite"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/${MODEL_NAME}"
MODEL_SIZE_EXPECTED="20M"  # 预期大小约 20MB

# 参数处理
MODEL_DIR="${1:-$HOME/lelamp_runtime/models}"
MODEL_FILE="$MODEL_DIR/$MODEL_NAME"

echo "======================================"
echo "  LeLamp 边缘视觉模型下载工具"
echo "======================================"
echo ""

# 检查目录
echo "📁 模型目录: $MODEL_DIR"
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${YELLOW}创建模型目录...${NC}"
    mkdir -p "$MODEL_DIR"
fi

# 检查是否已存在
if [ -f "$MODEL_FILE" ]; then
    EXISTING_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo -e "${YELLOW}⚠️  模型文件已存在: $MODEL_FILE ($EXISTING_SIZE)${NC}"
    echo ""
    read -p "是否重新下载？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "取消下载"
        exit 0
    fi
    rm -f "$MODEL_FILE"
fi

# 下载模型
echo ""
echo "📥 正在下载 $MODEL_NAME ..."
echo "   来源: $MODEL_URL"
echo "   预期大小: ~$MODEL_SIZE_EXPECTED"
echo ""

# 使用 curl 下载，显示进度条
if curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL"; then
    echo ""
    # 验证下载
    if [ -f "$MODEL_FILE" ]; then
        # macOS 和 Linux 的 du 命令参数不同
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            ACTUAL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
            ACTUAL_BYTES=$(stat -f%z "$MODEL_FILE")
        else
            # Linux
            ACTUAL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
            ACTUAL_BYTES=$(du -b "$MODEL_FILE" | cut -f1)
        fi

        echo ""
        echo -e "${GREEN}✅ 模型下载成功！${NC}"
        echo "   文件路径: $MODEL_FILE"
        echo "   文件大小: $ACTUAL_SIZE"

        # 检查文件大小是否合理
        if [ "$ACTUAL_BYTES" -lt 10000000 ]; then
            echo ""
            echo -e "${RED}❌ 警告: 文件大小异常 ($ACTUAL_BYTES < 10MB)${NC}"
            echo "   文件可能损坏或不完整，请重新下载"
            exit 1
        fi

        # 设置权限
        chmod 644 "$MODEL_FILE"

        echo ""
        echo "📝 下一步:"
        echo "   1. 在 .env 中设置: LELAMP_EDGE_VISION_ENABLED=true"
        echo "   2. 配置模型路径: LELAMP_EDGE_VISION_MODEL_DIR=$MODEL_DIR"
        echo "   3. 重启 LeLamp 服务"
        echo ""
        echo "📖 详细配置指南: docs/EDGE_VISION_SETUP.md"

    else
        echo -e "${RED}❌ 下载失败: 文件不存在${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ 下载失败${NC}"
    echo ""
    echo "可能的原因:"
    echo "  1. 网络连接问题"
    echo "  2. Google Storage 访问受限"
    echo "  3. 磁盘空间不足"
    echo ""
    echo "解决方案:"
    echo "  - 检查网络连接"
    echo "  - 使用 VPN 或代理"
    echo "  - 手动下载: $MODEL_URL"
    echo "  - 参考文档: docs/EDGE_VISION_SETUP.md"
    exit 1
fi

echo ""
echo "======================================"
echo "  下载完成"
echo "======================================"
