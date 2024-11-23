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
fov_h = np.radians(87)  # D455의 가로 FOV
fov_v = np.radians(58)  # D455의 세로 FOV
pixel_area = (np.tan(fov_h / 2) * 2 / width) * (np.tan(fov_v / 2) * 2 / height)

# 부피 계산 함수
def calculate_volume_integral(depth_frame, mask, pixel_area):
    mask_indices = np.where(mask > 0)
    depths = depth_frame[mask_indices]
    non_zero_depths = depths[depths > 0]
    if len(non_zero_depths) == 0:
        return 0
    total_volume = np.sum(non_zero_depths * pixel_area)
    return total_volume

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # YOLO Segmentation 모델 실행
        results = model(color_image)

        for result in results:
            if result.masks is not None:  # Segmentation 결과가 있을 경우 처리
                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                class_names = model.names  # 클래스 이름 가져오기

                for i, mask in enumerate(masks):
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if len(contours) == 0:
                        continue

                    # 부피 계산
                    volume = calculate_volume_integral(depth_image, binary_mask, pixel_area)
                    print(f"Object {i+1} - Class: {class_names[int(classes[i])]} - Volume: {volume:.3f} m³")

                    # 결과 시각화
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(color_image, f"Vol: {volume:.2f} m³", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    color_image[binary_mask > 0] = [0, 0, 255]

        # 시각
        cv2.imshow("YOLO Segmentation Results", color_image)
        cv2.imshow("Depth Colormap", cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
