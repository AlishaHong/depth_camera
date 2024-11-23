# !pip install pyrealsense2

import pyrealsense2 as rs  # Intel RealSense 카메라의 스트리밍 데이터를 처리하기 위한 라이브러리
import numpy as np  # 배열 연산 및 데이터 처리 라이브러리
import cv2  
from ultralytics import YOLO

# YOLO Segmentation 모델 로드
# model_path = "best.pt"  # YOLO Segmentation 모델 파일 경로(학습된 모델)
# model = YOLO(model_path, verbose=False)  # YOLO 모델 객체 생성, verbose=False로 불필요한 로그 출력 방지

# 일단 사전학습된 모델로 확인해보기
model = YOLO('yolo11n-seg.pt') 

# RealSense 파이프라인 초기화
pipeline = rs.pipeline()  # RealSense 카메라 스트리밍을 처리하는 파이프라인 객체 생성
config = rs.config()  # RealSense 카메라 설정 객체 생성

# 깊이 스트림 활성화 (해상도: 640x480, 데이터 포맷: z16, 프레임 속도: 30fps)
# 데이터 포맷 : z16 16비트 정수 형식으로 깊이 데이터를 저장(각 픽셀은 mm 단위)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 컬러 스트림 활성화 (해상도: 640x480, 데이터 포맷: BGR, 프레임 속도: 30fps)
# 데이터 포맷 : bgr8 8비트 정수 형식으로 BGR 순서의 데이터를 저장
# opencv와 동일한 포맷으로 저장
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 설정된 스트리밍 시작
pipeline.start(config)  # RealSense 카메라 스트리밍 시작


# 아래의 함수는 높이의 평균을 제공하여 부피를 계산하게 한다
# 이 방법은 간단하고 빠르게 부피를 측정할 수 있다는 장점이 있지만
# 얼마 전 원경이에게 들었던 적분하여 부피를 계산하는 방식이 더 정확할 수 있다고 한다(지피티)

# 픽셀별 부피를 계산하는 함수 정의
def calculate_volume_integral(depth_frame, mask, pixel_area):
    
    # 적분 방식으로 부피를 계산하는 함수.
    # param depth_frame: RealSense에서 가져온 깊이 데이터 (2D 배열 형태)
    # param mask: YOLO Segmentation 모델에서 추출한 객체의 마스크 (이진화된 배열)
    # param pixel_area: 각 픽셀의 실제 면적 (단위: m²)
    # return: 적분 기반으로 계산된 부피 값 (단위: m³)
 
    mask_indices = np.where(mask > 0)  # 마스크에서 객체가 포함된 픽셀 위치를 가져옴
    depths = depth_frame[mask_indices]  # 해당 픽셀 위치의 깊이 값을 가져옴
    non_zero_depths = depths[depths > 0]  # 깊이가 0이 아닌 값만 필터링
    
    if len(non_zero_depths) == 0:  # 유효한 깊이 값이 없으면 0 반환
        return 0

    # 각 픽셀의 깊이에 픽셀 면적을 곱하여 전체 부피를 계산
    total_volume = np.sum(non_zero_depths * pixel_area)
    return total_volume  # 계산된 부피 반환




try:
    while True:  # 무한 루프 시작 (실시간 스트리밍 처리)
        # RealSense 카메라에서 프레임 데이터를 가져옴
        frames = pipeline.wait_for_frames()  # 깊이와 컬러 데이터를 동기화하여 가져옴

        # 깊이 프레임과 컬러 프레임 분리
        depth_frame = frames.get_depth_frame()  # 깊이 데이터 프레임 가져오기
        color_frame = frames.get_color_frame()  # 컬러 이미지 프레임 가져오기

        # 유효하지 않은 프레임이면 다음 반복으로 건너뜀
        if not depth_frame or not color_frame:
            continue

        # 컬러와 깊이 데이터를 NumPy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())  # 컬러 이미지를 NumPy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())  # 깊이 데이터를 NumPy 배열로 변환

        # 픽셀 면적 계산 (카메라 FOV와 해상도를 기반으로 계산)
        # 예: RealSense D435의 FOV가 가로 87°, 세로 58°인 경우
        # D455도 시야각 동일함
        # fov는 하드웨어마다 정해진 값이 있다. 사용하는 모델의 스펙을 확인해 볼것
        width, height = 640, 480  # 카메라 해상도
        fov_h = np.radians(87)  # 가로 FOV를 라디안으로 변환
        fov_v = np.radians(58)  # 세로 FOV를 라디안으로 변환
        pixel_area = (np.tan(fov_h / 2) * 2 / width) * (np.tan(fov_v / 2) * 2 / height)  # 픽셀 면적 계산

        # YOLO Segmentation 모델을 사용하여 객체 탐지 수행
        results = model(color_image)  # 컬러 이미지를 YOLO 모델에 입력하여 탐지 실행

        # YOLO 탐지 결과 반복 처리
        for result in results:  # 탐지된 결과 객체 반복
            if result.segmentation is not None:  # Segmentation 결과가 있을 경우에만 처리
                masks = result.masks.data.cpu().numpy()  # Segmentation 마스크 데이터 가져오기
                classes = result.boxes.cls.cpu().numpy()  # 탐지된 객체의 클래스 정보 가져오기

                # 탐지된 모든 객체에 대해 반복 처리
                for i, mask in enumerate(masks):
                    # 마스크 데이터를 이진화 (값이 0.5 이상인 부분을 1로 설정)
                    binary_mask = (mask > 0.5).astype(np.uint8)

                    # 마스크 영역에서 컨투어(외곽선) 찾기
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # 컨투어가 없으면 건너뜀 (유효 객체가 없다는 의미)
                    if len(contours) == 0:
                        continue

                    # 적분 방식으로 부피 계산
                    volume = calculate_volume_integral(depth_image, binary_mask, pixel_area)

                    print(f"Object {i+1} - Class: {classes[i]} - Volume: {volume:.3f} m³")

                    # 결과 시각화 (경계 박스와 부피 값 표시)
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 경계 박스 그리기
                    cv2.putText(color_image, f"Vol: {volume:.2f} m³", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 부피 값 표시
                    color_image[binary_mask > 0] = [0, 0, 255]  # 마스크 영역 강조

        # YOLO Segmentation 결과를 OpenCV 창으로 시각화
        cv2.imshow("YOLO Segmentation Results", color_image)  # 컬러 이미지에 Segmentation 결과 표시
        cv2.imshow("Depth Colormap", cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))  # 깊이 데이터 시각화

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 스트리밍 종료 및 자원 해제
    pipeline.stop()  # RealSense 파이프라인 종료
    cv2.destroyAllWindows()  # OpenCV 창 닫기
