# 거리 -> 높이 코드에서
# 1. 픽셀 면적을 구하는 부분과
# 2. 높이값에 픽셀면적을 곱해서 누적합을 구하는 부분만 추가됨 

# 픽셀 면적을 구하는 부분의 수식이 어렵다. 필요한건 해상도와 시야각 정보라는건 알겠는데 저 수식에 대한 이해가 필요

# [픽셀면적?]
# 픽셀 면적이란 카메라로 찍힌 이미지에서 하나의 픽셀이 실제 세상의 어느정도의 면적을 나타내는지를 뜻함
# 픽셀면적은 카메라의 시야각과 해상도에 따라 달라진다.

# 카메라에서 특정 물체의 크기(넓이)를 측정하려면, 물체가 몇 픽셀인지 알아야하고
# 한 픽셀이 실제로 얼마나 넓은지를 알아야 물체의 크기를 계산할 수 있다.

# [수식 이해하기]
# tan : 삼각함수 "각도에 따른 세로길이와 가로길이의 비율"을 알려줌
# pixel_area = (np.tan(fov_h / 2) * 2 / width) * (np.tan(fov_v / 2) * 2 / height)
# fov_h 값을 반으로 나누어 삼각함수에 넣으면 카메라 중심에서 시야 끝까지의 거리 비율을 구할수 있음
# 반으로 나누는 이유 계산의 단순화를 위함
# 아무튼 그렇게 구해진 시야의 범위가 예를들어 가로 2.0 세로 1.5라고 치면 
# 각각의 가로 세로 픽셀로 나눈다음 둘을 곱하면 한 픽셀의 면적을 구할 수 있다. 
# 하나하나 계산된 픽셀의 면적에 각각의 높이를 곱하면 부피 완성! 

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

# 픽셀 면적 계산
width, height = 640, 480
fov_h = np.radians(87)  # 가로 FOV (D455 기준)
fov_v = np.radians(58)  # 세로 FOV
pixel_area = (np.tan(fov_h / 2) * 2 / width) * (np.tan(fov_v / 2) * 2 / height)  # 각 픽셀 면적 (m²)

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