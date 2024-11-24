# 높이를 구하는 방법 

# 앞 코드에서 카메라와 객체간의 거리를 구해봤다. 
# 높이는 코드를 조금만 추가하면 된다.
# 1. 카메라와 바닥 사이의 거리를 상수 a로 지정해둔다.
# 2. segmentation으로 탐지된 특정 객체의 모든 픽셀의 각 거리를 
# 3. a에서 빼주면 그 값이 각 픽셀의 높이가 된다. 
# 4. 다음 부피코드에서 각 픽셀의 높이에 픽셀면적값을 곱하여 부피값을 구할 예정

# 나머지는 1-1 코드를 참고 



import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# YOLO Segmentation 모델 로드
model = YOLO('yolo11n-seg.pt')  # 학습된 YOLO Segmentation 모델 사용

# RealSense 파이프라인 초기화
pipeline = rs.pipeline()  # RealSense 카메라 스트리밍 데이터를 처리하는 파이프라인 생성
config = rs.config()  # RealSense 카메라 설정 객체 생성

# 깊이 데이터 스트림 활성화 (해상도: 640x480, 프레임 속도: 30fps)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 컬러 데이터 스트림 활성화 (BGR 포맷, 해상도: 640x480, 프레임 속도: 30fps)
# BGR은 OpenCV에서 기본적으로 사용하는 색상 순서로, RGB 데이터를 OpenCV와 바로 호환 가능하게 처리
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 파이프라인 시작 (설정된 스트림으로 데이터 수신 시작)
pipeline.start(config)

# 고정된 바닥 거리 (카메라와 바닥 사이의 거리, 단위: m)
fixed_floor_distance = 2.0  # 카메라와 바닥 간의 고정 거리 (2m로 가정)

try:
    while True:  # 실시간 데이터 처리를 위해 무한 루프 실행
        # RealSense 카메라로부터 깊이와 컬러 프레임 가져오기
        frames = pipeline.wait_for_frames()  # 깊이 및 컬러 데이터를 동기화하여 수신
        depth_frame = frames.get_depth_frame()  # 깊이 데이터 프레임 가져오기
        color_frame = frames.get_color_frame()  # 컬러 이미지 프레임 가져오기

        if not depth_frame or not color_frame:  # 유효하지 않은 프레임이면 루프 건너뜀
            continue

        # 컬러 및 깊이 데이터를 NumPy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())  # 컬러 이미지를 NumPy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())  # 깊이 이미지를 NumPy 배열로 변환

        # YOLO Segmentation 모델 실행 (객체 탐지 및 Segmentation 수행)
        results = model(color_image)

        for result in results:  # 탐지 결과를 반복하여 처리
            if result.masks is not None:  # Segmentation 결과가 있을 경우 처리
                # Segmentation 마스크 데이터 및 클래스 정보를 NumPy 배열로 변환
                masks = result.masks.data.cpu().numpy()  # 마스크 데이터 (각 객체의 영역 표시)
                classes = result.boxes.cls.cpu().numpy()  # 탐지된 객체의 클래스 ID
                class_names = model.names  # 클래스 ID를 이름으로 변환하기 위한 매핑

                for i, mask in enumerate(masks):  # 탐지된 각 객체에 대해 마스크 처리
                    # Segmentation 마스크를 이진화 (객체 영역을 1, 나머지를 0으로 설정)
                    binary_mask = (mask > 0.5).astype(np.uint8)

                    # 객체의 모든 픽셀 좌표를 가져옴 (y, x 형태로 반환)
                    mask_indices = np.where(binary_mask > 0)

                    # 해당 좌표의 깊이 값을 가져옴
                    object_depths = depth_image[mask_indices]

                    # 유효한 깊이 값 필터링 (깊이가 0보다 큰 값만 사용)
                    object_depths = object_depths[object_depths > 0]

                    if len(object_depths) == 0:  # 유효한 깊이 값이 없으면 해당 객체 건너뜀
                        continue

                    # 각 픽셀의 높이를 계산 (카메라-바닥 거리에서 객체 깊이를 뺌)
                    heights = fixed_floor_distance - object_depths  #부피구할땐 얘만 필요
                    heights = np.maximum(heights, 0)  # 음수 값 방지 (높이는 0 이상이어야 함)

                    # 객체 높이의 최소, 최대, 평균 값 계산(부피구할땐 필요없엉)
                    min_height = np.min(heights)  # 최소 높이
                    max_height = np.max(heights)  # 최대 높이
                    avg_height = np.mean(heights)  # 평균 높이

                    # 객체의 중심점 계산 (Bounding Box 사용)
                    x, y, w, h = cv2.boundingRect(binary_mask)  # 객체의 경계 상자 계산
                    cx, cy = x + w // 2, y + h // 2  # 경계 상자의 중심 좌표 계산

                    # 결과를 콘솔에 출력
                    print(f"Object {i+1} - Class: {class_names[int(classes[i])]} "
                          f"- Min Height: {min_height:.3f} m, Max Height: {max_height:.3f} m, Avg Height: {avg_height:.3f} m")

                    # 결과 시각화
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 경계 상자 그리기
                    cv2.putText(color_image, f"Min: {min_height:.2f}m, Max: {max_height:.2f}m", 
                                (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 최소, 최대 높이 표시
                    cv2.putText(color_image, f"Avg: {avg_height:.2f}m", 
                                (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 평균 높이 표시

        # 시각화: YOLO Segmentation 결과를 컬러 이미지에 표시
        cv2.imshow("YOLO Segmentation Results", color_image)

        # 시각화: 깊이 데이터를 컬러맵으로 변환하여 표시 (HSV로 깊이 데이터를 시각화)
        cv2.imshow("Depth Colormap", cv2.applyColorMap(cv2.convertScaleAbs(depth_image * 8, alpha=0.03), cv2.COLORMAP_HSV))

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # RealSense 파이프라인 및 OpenCV 자원 해제
    pipeline.stop()
    cv2.destroyAllWindows()
