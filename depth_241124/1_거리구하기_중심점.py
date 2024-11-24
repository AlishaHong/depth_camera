import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# YOLO Segmentation 모델 로드
model = YOLO('yolo11n-seg.pt')  # 학습된 YOLO Segmentation 모델 사용

# RealSense 파이프라인 초기화
pipeline = rs.pipeline()    # RealSense 카메라 스트리밍 데이터를 처리하는 파이프라인 생성
# config - 카메라의 데이터스트림(깊이, 컬러등)을 설정하는데 사용
config = rs.config()    # RealSense 카메라 설정 객체 생성
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 대부분 rgb로 받아서 수정해주는데 바로 bgr포맷 사용
pipeline.start(config)  # RealSense 카메라 스트리밍 시작

try:
    while True: # 카메라에서 데이터를 실시간으로 처리하기 위해 무한루프실행(웹캠과 같음)
        frames = pipeline.wait_for_frames() # 깊이와 컬러 데이터를 동기화하여 가져옴
        depth_frame = frames.get_depth_frame()  # 깊이 데이터 프레임 가져오기
        color_frame = frames.get_color_frame()  # 컬러 이미지 프레임 가져오기

        if not depth_frame or not color_frame:
            continue    # 유효하지 않은 프레임이면 다음 루프로 건너뜀

        # 컬러데이터(이미지), 깊이데이터(이미지)를 넘파이 배열로 변환(opencv와 yolo와 호환되도록 해야함)
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # YOLO Segmentation 모델 실행   
        results = model(color_image)    # YOLO 모델을 사용하여 컬러 이미지에서 객체 탐지 수행

        for result in results:  # 탐지를 반복하여 처리
            if result.masks is not None:  # Segmentation 결과가 있을 경우 처리
                masks = result.masks.data.cpu().numpy() # Segmentation 마스크 데이터를 NumPy 배열로 변환
                classes = result.boxes.cls.cpu().numpy()    # 탐지된 객체의 클래스 정보 가져오기(id)
                class_names = model.names  # 클래스 이름 가져오기(클래스id를 이름으로 변환)

                for i, mask in enumerate(masks):    # numpy로 변환된 seg마스크 데이터에 인덱스 짝꿍 만들어주기
                    binary_mask = (mask > 0.5).astype(np.uint8) # 마스크(넘파이배열)를 이진화
                    # 마스크데이터(원본 numpy 배열 : 0~1범위의 실수)
                    # mask = np.array([[0.2, 0.6, 0.4],
                    #                 [0.8, 0.3, 0.7],
                    #                 [0.1, 0.9, 0.5]])
                    
                    # 이진화 조건 적용(mask > 0.5) : 객체가 있는 영역과 없는 영역을 구분
                    # mask > 0.5
                    # 결과 (논리 배열):
                    # array([[False,  True, False],
                    #     [ True, False,  True],
                    #     [False,  True, False]])
                    
                    # 정수형 변환(astpye(np.uint8))
                    # array([[0, 1, 0],
                    #     [1, 0, 1],
                    #     [0, 1, 0]], dtype=uint8)
                    
                    # cv2.findContours는 픽셀 값이 0 또는 1이어야 경계선을 올바르게 찾는다.
                    # 이진화된 이미지를 입력으로 받아, 객체의 경계를 찾는 OpenCV 함수
                    
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if len(contours) == 0:
                        continue

                    # 객체의 중심점 계산
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))   # 가장 큰 컨투어의 경계상자 계산
                    cx, cy = x + w // 2, y + h // 2

                    # 중심점에서의 거리 (깊이 값)
                    distance = depth_frame.get_distance(cx, cy)
                    print(f"Object {i+1} - Class: {class_names[int(classes[i])]} - Distance: {distance:.3f} m")

                    # 결과 시각화
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(color_image, f"Dist: {distance:.2f} m", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 시각화
        cv2.imshow("YOLO Segmentation Results", color_image)
        cv2.imshow("Depth Colormap", cv2.applyColorMap(cv2.convertScaleAbs(depth_image * 8, alpha=0.03), cv2.COLORMAP_HSV))
        # 거리감의 시각화를 더욱 분명하게 하기위해 depth_image에 * 8 해주고 컬러맵도 변경함 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()