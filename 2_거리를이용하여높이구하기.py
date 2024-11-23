import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# YOLO Segmentation 모델 로드
model = YOLO('yolo11n-seg.pt')  # 학습된 YOLO Segmentation 모델 사용

# RealSense 파이프라인 초기화
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 고정된 바닥 거리 (카메라와 바닥 사이의 거리)
fixed_floor_distance = 2.0  # 단위: m

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 프레임을 NumPy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # YOLO Segmentation 모델 실행
        results = model(color_image)

        for result in results:
            if result.masks is not None:  # Segmentation 결과가 있을 경우 처리
                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                class_names = model.names

                for i, mask in enumerate(masks):
                    binary_mask = (mask > 0.5).astype(np.uint8)

                    # 객체에 해당하는 픽셀의 깊이 값 추출
                    mask_indices = np.where(binary_mask > 0)  # 객체에 해당하는 픽셀 좌표
                    object_depths = depth_image[mask_indices]  # 깊이 데이터에서 해당 좌표의 값 추출

                    # 유효한 깊이 값 필터링
                    object_depths = object_depths[object_depths > 0]

                    if len(object_depths) == 0:
                        continue

                    # 객체의 최소, 최대, 평균 높이 계산
                    heights = fixed_floor_distance - object_depths  # 높이 계산
                    min_height = np.min(heights)
                    max_height = np.max(heights)
                    avg_height = np.mean(heights)

                    # 객체의 중심점 계산
                    x, y, w, h = cv2.boundingRect(binary_mask)
                    cx, cy = x + w // 2, y + h // 2

                    print(f"Object {i+1} - Class: {class_names[int(classes[i])]} "
                          f"- Min Height: {min_height:.3f} m, Max Height: {max_height:.3f} m, Avg Height: {avg_height:.3f} m")

                    # 결과 시각화
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(color_image, f"Min: {min_height:.2f}m, Max: {max_height:.2f}m", 
                                (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(color_image, f"Avg: {avg_height:.2f}m", 
                                (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 시각화
        cv2.imshow("YOLO Segmentation Results", color_image)
        cv2.imshow("Depth Colormap", cv2.applyColorMap(cv2.convertScaleAbs(depth_image * 8, alpha=0.03), cv2.COLORMAP_HSV))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
