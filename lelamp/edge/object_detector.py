"""
MediaPipe 物体检测服务

用于本地物体识别，减少云端 API 调用。
"""
import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("lelamp.edge.object")

# MediaPipe 是可选依赖，优雅降级
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    import cv2
    # 检查是否有 solutions API 或 tasks API（新版 MediaPipe）
    if hasattr(mp, 'solutions'):
        MEDIAPIPE_AVAILABLE = True
    elif hasattr(mp, 'tasks'):
        MEDIAPIPE_AVAILABLE = True
    else:
        logger.warning("MediaPipe installed but required API not available. "
                      "ObjectDetector will run in NoOp mode.")
except ImportError:
    logger.warning("MediaPipe not available, ObjectDetector will run in NoOp mode")


@dataclass
class DetectedObject:
    """检测到的物体"""
    label: str           # 中文标签
    label_en: str        # 英文标签
    confidence: float    # 置信度
    bbox: List[int]      # [x, y, width, height]
    category_id: int     # COCO 类别 ID


class ObjectDetector:
    """
    基于 MediaPipe 的物体检测服务
    
    使用 EfficientDet-Lite 模型，支持 80 类 COCO 物体。
    本地推理，低延迟（100-300ms）。
    
    使用示例:
        detector = ObjectDetector()
        
        while True:
            ret, frame = cap.read()
            result = detector.detect(frame)
            if result["objects"]:
                print(f"检测到: {result['summary']}")
    """
    
    # COCO 类别中文名映射（常用物品）
    COCO_LABELS_ZH = {
        1: "人", 2: "自行车", 3: "汽车", 4: "摩托车", 5: "飞机",
        6: "公交车", 7: "火车", 8: "卡车", 9: "船",
        10: "红绿灯", 11: "消防栓", 13: "停车标志",
        14: "停车计时器", 15: "长凳",
        16: "猫", 17: "狗", 18: "马", 19: "羊", 20: "牛",
        21: "大象", 22: "熊", 23: "斑马", 24: "长颈鹿",
        25: "背包", 26: "雨伞", 27: "手提包", 28: "领带",
        29: "行李箱", 31: "钱包",
        33: "运动球", 34: "风筝", 35: "棒球棒",
        36: "棒球手套", 37: "滑板", 38: "冲浪板",
        39: "网球拍", 41: "滑冰鞋",
        42: "杯子", 43: "叉子", 44: "刀", 45: "勺子", 46: "碗",
        47: "香蕉", 48: "苹果", 49: "三明治", 50: "橙子",
        51: "西兰花", 52: "胡萝卜", 53: "热狗", 54: "披萨",
        55: "甜甜圈", 56: "蛋糕",
        57: "椅子", 58: "沙发", 59: "盆栽", 60: "床",
        61: "餐桌", 62: "厕所", 63: "电视", 64: "笔记本电脑",
        65: "鼠标", 66: "遥控器", 67: "键盘", 68: "手机",
        69: "微波炉", 70: "烤箱", 71: "烤面包机", 72: "水槽",
        73: "冰箱", 74: "书", 75: "时钟", 76: "花瓶",
        77: "剪刀", 78: "泰迪熊", 79: "吹风机", 80: "牙刷",
    }
    
    # 常见物品简化映射（用于快速回答）
    COMMON_OBJECTS = {
        "苹果": "水果",
        "香蕉": "水果",
        "杯子": "容器",
        "书": "学习用品",
        "手机": "电子产品",
        "键盘": "电子产品",
        "鼠标": "电子产品",
        "笔记本电脑": "电子产品",
        "电视": "电子产品",
        "椅子": "家具",
        "沙发": "家具",
        "桌子": "家具",
        "猫": "宠物",
        "狗": "宠物",
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        max_results: int = 5,
        score_threshold: float = 0.5,
    ):
        """
        初始化物体检测器
        
        Args:
            model_path: 模型文件路径，默认使用内置模型
            max_results: 最大返回结果数
            score_threshold: 置信度阈值
        """
        self._noop = not MEDIAPIPE_AVAILABLE
        self._max_results = max_results
        self._score_threshold = score_threshold
        self._model_path = model_path
        
        if not self._noop:
            self._init_detector()
        
        # 统计信息
        self._total_detections = 0
        self._object_counts: Dict[str, int] = {}
        
        mode = "NoOp" if self._noop else "MediaPipe"
        logger.info(f"ObjectDetector initialized ({mode} mode)")
    
    def _init_detector(self):
        """初始化 MediaPipe 物体检测器"""
        try:
            # 尝试使用新版的 tasks API
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # 如果没有指定模型路径，使用默认模型
            if self._model_path is None:
                self._model_path = self._get_default_model_path()
            
            if self._model_path and os.path.exists(self._model_path):
                base_options = python.BaseOptions(model_asset_path=self._model_path)
                options = vision.ObjectDetectorOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    max_results=self._max_results,
                    score_threshold=self._score_threshold,
                )
                self.detector = vision.ObjectDetector.create_from_options(options)
                self._use_tasks_api = True
                logger.info(f"ObjectDetector using tasks API with model: {self._model_path}")
            else:
                # 降级到旧版 API（不支持物体检测）
                logger.warning("Model file not found, ObjectDetector will run in limited mode")
                self._noop = True
                
        except ImportError:
            # 旧版 MediaPipe 不支持 tasks API
            logger.warning("MediaPipe tasks API not available, ObjectDetector will run in NoOp mode")
            self._noop = True
        except Exception as e:
            logger.error(f"Failed to initialize ObjectDetector: {e}")
            self._noop = True
    
    def _get_default_model_path(self) -> Optional[str]:
        """获取默认模型路径"""
        # 尝试几个常见的模型位置
        possible_paths = [
            # 项目目录
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "efficientdet_lite0.tflite"),
            # 用户主目录
            os.path.expanduser("~/.mediapipe/models/efficientdet_lite0.tflite"),
            # /tmp 目录
            "/tmp/mediapipe/efficientdet_lite0.tflite",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 如果模型不存在，提示用户下载
        logger.info(
            "Object detection model not found. "
            "Download from: https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
        )
        return None
    
    def detect(self, frame) -> Dict[str, Any]:
        """
        检测帧中的物体
        
        Args:
            frame: BGR 格式的图像帧
            
        Returns:
            {
                "objects": [DetectedObject, ...],
                "labels": ["苹果", "杯子"],
                "summary": "检测到：苹果、杯子",
                "count": int
            }
        """
        if self._noop:
            return self._noop_detect()
        
        try:
            # 转换为 RGB
            import cv2
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 创建 MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # 执行检测
            results = self.detector.detect(mp_image)
            
            objects: List[DetectedObject] = []
            labels: List[str] = []
            
            for detection in results.detections:
                category = detection.categories[0]
                label_en = category.category_name
                category_id = category.index
                confidence = category.score
                bbox = detection.bounding_box
                
                label_zh = self.COCO_LABELS_ZH.get(category_id, label_en)
                
                objects.append(DetectedObject(
                    label=label_zh,
                    label_en=label_en,
                    confidence=confidence,
                    bbox=[bbox.origin_x, bbox.origin_y, bbox.width, bbox.height],
                    category_id=category_id
                ))
                
                if label_zh not in labels:
                    labels.append(label_zh)
                
                # 更新统计
                self._object_counts[label_zh] = self._object_counts.get(label_zh, 0) + 1
            
            self._total_detections += 1
            
            summary = f"检测到：{', '.join(labels)}" if labels else "未检测到物体"
            
            return {
                "objects": objects,
                "labels": labels,
                "summary": summary,
                "count": len(objects)
            }
            
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return self._noop_detect()
    
    def _noop_detect(self) -> Dict[str, Any]:
        """NoOp 模式的默认返回"""
        return {
            "objects": [],
            "labels": [],
            "summary": "物体检测服务未启用",
            "count": 0
        }
    
    def get_category(self, label: str) -> str:
        """获取物体的分类"""
        return self.COMMON_OBJECTS.get(label, "物品")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_detections": self._total_detections,
            "object_counts": self._object_counts.copy(),
            "noop_mode": self._noop
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self._total_detections = 0
        self._object_counts = {}
    
    def close(self):
        """释放资源"""
        if not self._noop and hasattr(self, 'detector'):
            self.detector.close()
            logger.info("ObjectDetector closed")