# [코드 이해해보기] 
# 이 코드를 이해하면 높이를 구하고 부피를 구하는건 계산만 하면 된다!
# 1_거리구하기_중심점 코드에서는 탐지된 객체의 중심점을 직접 구해서 
# 그 중심점과 카메라 사이의 거리를 출력하도록 했다.
# 쉬운 방법이지만 굴곡이 있는 음식데이터에는 적합하지 않아서 
# 각각의 픽셀값의 높이를 구해서 전체 부피를 구하는 코드를 사용할 예정

# [가장 중요한것]
# 각각의 픽셀값을 정의하고 그 각각의 픽셀값까지의 거리들을 구하는것이 관건!!!!! 

# 중심점 코드와 다른건 몇줄 되지 않는다. 
# 마스크(segmentation객체)내에서 유효한(뭔가가 있는! 값이 0.5이상인) 픽셀값을 나열하고 
# 해당 지점에서의 깊이 값을 가져오면 되는데 
# 깊이 값을 어떻게 가져오느냐! get_data()라는 함수를 이용한다. 



# [순서대로 살펴보기]
# 1. 최초에 리얼센스 파이프라인을 초기화해주고 
# 2. 카메라 설정하는 객체도 생성해주고
# 3. 카메라 스트리밍을 시작해주고 실시간 처리를 위해 while문 돌려서 frame값을 가져오는건 
# 이전에 했던 웹캠테스트와 매우 흡사하고 
# 다른점은 frame값이 '색상 정보', '깊이 정보' 두가지를 준다는 점이다.(렌즈가 두개니깐 > <)
# (get_depth_frame(), get_color_frame())
# frame은 RealSense 카메라로부터 가져온 데이터를 뜻하며,
# 이 프레임은 카메라가 캡처한 2D 배열 형태이다! 

# 4. 2d 배열 형태를 opencv나 yolo와 호환되도록 하기 위해서 넘파이 배열로 변환해야하는데 
# 각 프레임에서 get_data()함수를 이용해서 가져오는데 원시데이터라 바로 사용하기 어렵고
# (메모리 주소값형태 - 정확히는 우리 얼마전 얘기한 포인터 ㅎㅎ)
# 넘파이 배열로 변경해줘야하는것 

# 5. 넘파이 배열로 변경된 color_image 값은 yolo segmentation을 하는데 제공되고
# 똑같이 넘파이 배열로 변경된 depth_image는 이 코드에서는 각각의 깊이 데이터를 가져오는데도 사용되지만
# 이전 코드(중심점)에서는 시각화하는데에서만 사용한다. 

# 6. 실시간으로 들어오는 이미지에 대해서 for문 반복을 하고(큰 반복) 3d 형태의 mask 정보를 반환한다. ex - (3,480,640) 객체가 3개인 이미지
# 7. 각 이미지에서 탐지된 여러 객체들에 대해서 for문 반복을 한다.(작은반복)
# 1차로 값이 0.5 이상인 부분을 필터링 하고 정수형으로 변환해 준뒤 

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# YOLO Segmentation 모델 로드
model = YOLO('yolo11n-seg.pt')  # 학습된 YOLO Segmentation 모델 사용

# RealSense 파이프라인 초기화
pipeline = rs.pipeline()  # RealSense 카메라 스트리밍 데이터를 처리하는 파이프라인 생성
# config - 카메라의 데이터 스트림(깊이, 컬러 등)을 설정하는데 사용
config = rs.config()  # RealSense 카메라 설정 객체 생성
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 깊이 스트림 활성화 (해상도: 640x480, 30fps)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 컬러 스트림 활성화 (BGR 포맷, OpenCV와 호환)
pipeline.start(config)  # RealSense 카메라 스트리밍 시작

try:
    while True:  # 카메라에서 데이터를 실시간으로 처리하기 위해 무한 루프 실행(웹캠과 비슷한 동작)
        frames = pipeline.wait_for_frames()  # 깊이와 컬러 데이터를 동기화하여 가져옴
        depth_frame = frames.get_depth_frame()  # 깊이 데이터 프레임 가져오기
        color_frame = frames.get_color_frame()  # 컬러 이미지 프레임 가져오기

        # 깊이 또는 컬러 프레임이 유효하지 않으면 루프 건너뜀
        if not depth_frame or not color_frame:
            continue

        # 컬러 데이터(이미지), 깊이 데이터(이미지)를 NumPy 배열로 변환(OpenCV와 YOLO와 호환되도록 처리)
        color_image = np.asanyarray(color_frame.get_data())  # 컬러 프레임 데이터를 NumPy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())  # 깊이 프레임 데이터를 NumPy 배열로 변환

        # YOLO Segmentation 모델 실행
        results = model(color_image)  # YOLO 모델을 사용하여 컬러 이미지에서 객체 탐지 수행

        for result in results:  # 탐지 결과를 반복하여 처리(여러 이미지(프레임)에 대한 yolo 결과반복)
            if result.masks is not None:  # Segmentation 결과가 있을 경우에만 처리
                masks = result.masks.data.cpu().numpy()  # Segmentation 마스크 데이터를 NumPy 배열로 변환
                # result.masks.data : segmentation 결과에서 마스크 데이터만 추출 
                # 이 데이터는 gpu에서 실행된 모델이 반환하기 때문에 gpu에 저장되어 있으므로
                # cpu로 옮겨와야 해 왜 가져오느냐? python 넘파이나 opencv에서 사용해야하니까!
                # 마스크 데이터의 형태 알아보기 
                # masks.shape = ex(2,480,640) 3d 배열
                # masks[0].shape = (480,640) 2d 배열
                # print(masks[0]) -> 확률값(0~1사이의 실수)

                classes = result.boxes.cls.cpu().numpy()  # 탐지된 객체의 클래스 정보(ID)를 NumPy 배열로 가져오기
                class_names = model.names  # 클래스 이름 가져오기 (ID를 이름으로 변환하기 위해 사용)
                # classes는 1끼 식사에서는 0~5 /  class_names은 01011001등 
                # 이 코드에서는 코코데이터셋을 사용하니 그 class내용이 출력되고! 

                # 탐지된 각 객체에 대해 Segmentation 마스크 처리
                for i, mask in enumerate(masks):  # NumPy로 변환된 Segmentation 마스크 데이터와 인덱스를 반복
                    binary_mask = (mask > 0.5).astype(np.uint8)  # 마스크를 이진화 (객체 영역을 1, 나머지를 0으로 설정)
                    # 예: 원본 마스크 (mask)는 0~1 사이의 실수 값을 가짐
                    # 이진화 결과:
                    # mask > 0.5 -> True/False -> 1/0 (객체가 있는 픽셀만 남김)

                    # 왜 이진화를 해야할까? 
                    # 1. 명확히 구분하기 위해서 애매하게 실수값으로 하지않고 있고 없고를 확실히! 
                    # 2. 객체가 있는 픽셀만 골라내는 필터링 작업 시 binary_mask를 통해 
                    # 정확하게 mask_indices 값을 골라내고 그값으로 깊이 데이터를 뽑아내는데 
                    # 깊이 추출 시 실수값인 masks값(원본)을 직접 사용한다면 여러단계의 처리로 불필요한 연산이 늘어나고
                    # 오류가 발생할 수 있다고 한다. 

                    # 객체에 해당하는 픽셀 좌표 가져오기
                    # indices는 index의 복수형/나에겐 어색한 단어인데.. 직관적인 변수가 생각나지 않았음
                    mask_indices = np.where(binary_mask > 0)  # 객체가 있는 픽셀 좌표를 가져옴 (y, x 형태로 반환)
                    object_depths = depth_image[mask_indices]  # 깊이 데이터에서 해당 좌표의 깊이 값 추출

                    # 유효한 깊이 값 필터링 (0 이상의 값만 사용)
                    object_depths = object_depths[object_depths > 0]

                    if len(object_depths) == 0:  # 유효한 깊이 값이 없으면 다음 객체로 건너뜀
                        continue

                    # 각 픽셀의 깊이를 사용하여 추가 분석 가능
                    print(f"Object {i+1} - Class: {class_names[int(classes[i])]} - Pixel Depths: {object_depths}")

                    # 픽셀 깊이의 평균 값 계산 (예: 객체 중심에서의 평균 거리 추정)
                    avg_depth = np.mean(object_depths)  # 평균 깊이 계산(일단 이걸 출력)
                    print(f"Object {i+1} - Average Depth: {avg_depth:.3f} m")

                    # 결과 시각화를 위해 객체의 Bounding Box 계산
                    x, y, w, h = cv2.boundingRect(binary_mask)  # 객체의 경계 상자를 계산
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 경계 상자 그리기
                    cv2.putText(color_image, f"Avg Depth: {avg_depth:.2f}m", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 평균 깊이 표시

        # 컬러 이미지와 깊이 데이터를 시각화
        cv2.imshow("YOLO Segmentation Results", color_image)  # YOLO Segmentation 결과를 컬러 이미지에 표시
        cv2.imshow("Depth Colormap", cv2.applyColorMap(cv2.convertScaleAbs(depth_image * 8, alpha=0.03), cv2.COLORMAP_HSV))
        # depth_image에 *8을 곱하여 깊이 값을 시각적으로 더 뚜렷하게 표현

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 루프 종료
            break

finally:
    pipeline.stop()  # RealSense 파이프라인 중지
    cv2.destroyAllWindows()  # OpenCV 창 닫기
