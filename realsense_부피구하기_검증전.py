# 검증 전 코드임
# !pip install pyrealsense2

import pyrealsense2 as rs  # Intel RealSense 카메라의 스트리밍 데이터를 처리하기 위한 라이브러리
import numpy as np  # 배열 연산 및 데이터 처리 라이브러리
import cv2  
from ultralytics import YOLO

# YOLO Segmentation 모델 로드
model_path = "best.pt"  # YOLO Segmentation 모델 파일 경로(학습된 모델)
model = YOLO(model_path, verbose=False)  # YOLO 모델 객체 생성, verbose=False로 불필요한 로그 출력 방지

# RealSense 파이프라인 초기화
pipeline = rs.pipeline()  # RealSense 카메라 스트리밍을 처리하는 파이프라인 객체 생성
config = rs.config()  # RealSense 카메라 설정 객체 생성

# 깊이 스트림 활성화 (해상도: 640x480, 데이터 포맷: z16, 프레임 속도: 30fps
# 데이터 포맷 : z16 16비트 정수형식으로 깊이 데이터를 저장(각 픽셀은 mm단위)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 컬러 스트림 활성화 (해상도: 640x480, 데이터 포맷: BGR, 프레임 속도: 30fps)
# 데이터 포맷 : bgr8 8비트 정수 형식으로 BGR 순서의 데이터를 저장
# opencv와 동일한 포맷으로 저장
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 설정된 스트리밍 시작
pipeline.start(config)  # RealSense 카메라 스트리밍 시작


# 아래의 함수는 높이의 평균을 제공하여 부피를 계산하게 한다 
# 이 방법은 간단하고 빠르게 부피를 측정할 수 있다는 장점이 있지만 
# 얼마전 원경이에게 들었던 적분하여 부피를 계산하는 방식이 더 정확할 수 있다고 한다(지피티)


# 깊이 계산 함수 정의
def calculate_average_depth(depth_frame, mask):
    
    # 특정 객체의 마스크를 기준으로 평균 깊이를 계산하는 함수.
    # depth_frame: RealSense에서 가져온 깊이 데이터 (2D 배열 형태)
    # mask: YOLO Segmentation 모델에서 추출한 객체의 마스크 (이진화된 배열)
    # return: 마스크 영역 내에서 유효 깊이 값의 평균
    
    mask_indices = np.where(mask > 0)  # 마스크에서 객체가 포함된 픽셀 위치를 가져옴
    depths = depth_frame[mask_indices]  # 해당 위치의 깊이 값을 가져옴
    non_zero_depths = depths[depths > 0]  # 깊이가 0이 아닌 (유효한) 값만 필터링
    if len(non_zero_depths) == 0:  # 유효한 깊이 값이 없으면 0 반환
        return 0
    return np.mean(non_zero_depths)  # 유효 깊이 값들의 평균 계산 후 반환

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

                    # 가장 큰 컨투어를 기준으로 경계 박스(Bounding Box) 생성
                    contour = max(contours, key=cv2.contourArea)  # 가장 큰 컨투어 선택
                    x, y, w, h = cv2.boundingRect(contour)  # 컨투어에서 경계 박스 좌표와 크기 추출

                    # 경계 박스 중심점에서 길이와 넓이를 계산
                    length_distance = depth_frame.get_distance(x + w // 2, y)  # 길이 계산
                    width_distance = depth_frame.get_distance(x, y + h // 2)  # 넓이 계산

                    # 높이는 마스크 영역의 평균 깊이를 기준으로 계산
                    height = calculate_average_depth(depth_image, binary_mask)

                    # 부피 계산 (길이 × 넓이 × 높이)
                    volume = length_distance * width_distance * height
                    print(f"Object {i+1} - Class: {classes[i]} - "
                          f"Length: {length_distance:.2f} m, Width: {width_distance:.2f} m, "
                          f"Height: {height:.2f} m, Volume: {volume:.3f} m³")

                    # 결과 시각화 (경계 박스와 부피 값 표시)
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