import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# YOLO Segmentation 모델 로드
model = YOLO('yolo11n-seg.pt')  # 학습된 YOLO Segmentation 모델 사용

# RealSense 파이프라인 초기화
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth 스트림 활성화
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color 스트림 활성화
pipeline.start(config)

# Color와 Depth의 Alignment 객체 생성
align_to = rs.stream.color  # Depth 이미지를 Color 이미지에 정렬
align = rs.align(align_to)

# 픽셀 면적 계산
width, height = 640, 480
fov_h = np.radians(87)  # 가로 FOV (D455 기준)
fov_v = np.radians(58)  # 세로 FOV
pixel_area = (np.tan(fov_h / 2) * 2 / width) * (np.tan(fov_v / 2) * 2 / height)  # 각 픽셀 면적 (m²)

# 고정된 바닥 거리 (카메라와 바닥 사이의 거리)
fixed_floor_distance = 2.0  # 단위: m

try:
    while True:
        frames = pipeline.wait_for_frames()  # 새로운 프레임을 수신
        aligned_frames = align.process(frames)  # Depth 이미지를 Color 이미지에 정렬

        # 정렬된 Depth와 Color 프레임 가져오기
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:  # 유효하지 않은 프레임 처리
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

                    # 각 픽셀 높이 계산
                    heights = fixed_floor_distance - object_depths
                    heights = np.maximum(heights, 0)  # 음수 방지

                    # 부피 계산(누적시킴)
                    volume = np.sum(heights * pixel_area)  # 부피 = 높이 × 픽셀 면적
                    print(f"Object {i+1} - Class: {class_names[int(classes[i])]} - Volume: {volume:.3f} m³")

                    # 결과 시각화
                    x, y, w, h = cv2.boundingRect(binary_mask)
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(color_image, f"Vol: {volume:.3f} m³", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 시각화
        cv2.imshow("YOLO Segmentation Results", color_image)
        cv2.imshow("Depth Colormap", cv2.applyColorMap(cv2.convertScaleAbs(depth_image * 8, alpha=0.03), cv2.COLORMAP_HSV))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
