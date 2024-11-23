# 파이썬에서 영상확인하는 코드- 웹캠 확인하듯 실시간 확인 가능
# 파이프라인 객체를 생성하여 프레임값을 받아온뒤 
# 깊이/컬러 프레임을 가져와 시각화 해볼 수 있는 코드(opencv 활용)

import pyrealsense2 as rs
import numpy as np
import cv2

# RealSense 파이프라인 초기화
pipeline = rs.pipeline()

# 파이프라인 시작
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 원하는 해상도와 프레임 레이트 설정
pipeline.start()


try:
    while True:
        # 프레임 가져오기
        frames = pipeline.wait_for_frames()
        
        # 컬러 프레임 가져오기
        color_frame = frames.get_color_frame()
        # 깊이 프레임 가져오기
        depth_frame = frames.get_depth_frame()
        
        
        # 유효하지 않은 프레임이면 건너뜀
        if not color_frame or not depth_frame:
            continue
        
        # opencv에서 시각화 처리 가능하도록 받아온 값을 numpy 배열로 변환함
        # 컬러 프레임 데이터를 NumPy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())
        # 깊이 프레임 데이터를 NumPy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # RGB에서 BGR로 변환
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        
        # 깊이 데이터를 컬러맵으로 변환
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image * 8, alpha=0.03), cv2.COLORMAP_HSV)
        
        #ValueError: all the input array dimensions except for the concatenation axis must match exactly, 
        # but along dimension 0, the array at index 0 has size 720 and the array at index 1 has size 480
        
        # depth_colormap을 color_image 크기에 맞춤
        depth_colormap_resized = cv2.resize(depth_colormap, (color_image.shape[1], color_image.shape[0]))
        # 두 프레임을 가로로 합침
        images = np.hstack((color_image, depth_colormap_resized))
        
        # OpenCV 창에 이미지 표시
        cv2.imshow('RealSense', images)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 파이프라인 정지
    pipeline.stop()
    cv2.destroyAllWindows()