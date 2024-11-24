# 깃헙에 있는 calibration을 실행시키면 
# 교정된 내용이 장치에 저장된다고 함 
# 해당 코드를 실행하여 교정시킨 후 아래의 코드로 시각적으로 검증해보자


import pyrealsense2 as rs
import numpy as np
import cv2

# RealSense 카메라 파이프라인 초기화
pipeline = rs.pipeline()  # RealSense 카메라 데이터를 처리하기 위한 파이프라인 생성
config = rs.config()  # 카메라 스트림 설정을 위한 Config 객체 생성

# Depth와 Color 스트림 활성화
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth 스트림 활성화 (해상도: 640x480, 30fps)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color 스트림 활성화 (BGR 포맷, OpenCV 호환)

# 카메라 스트림 시작
pipeline.start(config)

try:
    while True:  # 실시간 데이터를 처리하기 위해 무한 루프 실행
        # Depth와 Color 프레임을 동시에 가져옴
        frames = pipeline.wait_for_frames()  # Depth와 Color 데이터를 동기화하여 가져옴
        depth_frame = frames.get_depth_frame()  # Depth 프레임 가져오기
        color_frame = frames.get_color_frame()  # Color 프레임 가져오기

        # 만약 Depth 또는 Color 프레임이 유효하지 않다면 다음 루프로 건너뜀
        if not depth_frame or not color_frame:
            continue

        # 프레임 데이터를 NumPy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())  # Depth 프레임 데이터를 NumPy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())  # Color 프레임 데이터를 NumPy 배열로 변환

        # Depth 이미지를 시각화 가능한 컬러맵으로 변환
        # alpha=0.03은 Depth 값을 시각적으로 더 뚜렷하게 표현하기 위한 스케일링 값
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # Color 이미지와 Depth 컬러맵을 겹침 (Overlay 효과)
        # cv2.addWeighted 함수는 두 이미지를 투명도를 조절하며 겹치는 데 사용됨
        overlay = cv2.addWeighted(color_image, 0.7, depth_colormap, 0.3, 0)
        # 0.7: Color 이미지의 가중치, 0.3: Depth 컬러맵의 가중치, 0: 밝기 조정 값

        # 시각화: 각 이미지 창에 결과 표시
        cv2.imshow('Color Image', color_image)         # 원본 Color 이미지 표시
        cv2.imshow('Depth Colormap', depth_colormap)  # Depth 데이터를 컬러맵으로 변환하여 표시
        cv2.imshow('Overlay', overlay)                # Color 이미지와 Depth 컬러맵을 겹쳐 표시

        # 사용자 입력으로 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 루프 종료
            break

finally:
    # 사용이 끝난 후 파이프라인 정리
    pipeline.stop()  # RealSense 파이프라인 중지
    cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
