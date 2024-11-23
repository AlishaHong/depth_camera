import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.utils.checks import check_yaml
from ultralytics.utils import ROOT, yaml_load

# Load YOLO model and class information
model = YOLO('yolov8n.pt')  # YOLO 모델 로드
CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Detect device and its sensors
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = any(s.get_info(rs.camera_info.name) == 'RGB Camera' for s in device.sensors)

if not found_rgb:
    print("RGB Camera not found! Exiting.")
    exit(1)

# Configure streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Start the pipeline
try:
    print("Starting pipeline with config:")
    profile = pipeline.start(config)
    print("Pipeline started successfully!")
except RuntimeError as e:
    print(f"Pipeline start failed: {e}")
    exit(1)

# Align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # RGB에서 BGR로 변환
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # Depth image visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.075), cv2.COLORMAP_JET)

        # YOLO processing
        results = model(color_image, stream=True)
        class_ids, confidences, bboxes = [], [], []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf
                if confidence > 0.5:
                    xyxy = box.xyxy.tolist()[0]
                    bboxes.append(xyxy)
                    confidences.append(float(confidence))
                    class_ids.append(box.cls.tolist())

        result_boxes = cv2.dnn.NMSBoxes(bboxes, confidences, 0.25, 0.45)

        for i in range(len(bboxes)):
            if i in result_boxes:
                bbox = list(map(int, bboxes[i]))
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                label = str(CLASSES[int(class_ids[i][0])])
                
                # 특정 클래스 이름 확인 (예: 'box')
                if label == 'box':  # 필요한 클래스 이름으로 변경
                    depth = depth_frame.get_distance(cx, cy)  # 중심점의 거리
                    depth_text = f"{depth * 100:.2f}cm"
                    label = f"{label} {confidences[i]:.2f} {depth_text}"

                    color = tuple(map(int, colors[i % len(colors)]))

                    # Draw bounding box and label
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Draw center point on depth colormap
                    cv2.circle(depth_colormap, (cx, cy), 5, color, -1)

        # Display the images
        cv2.imshow("RGB Stream", color_image)
        cv2.imshow("Depth Stream", depth_colormap)

        # Exit on 'q' or ESC
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
