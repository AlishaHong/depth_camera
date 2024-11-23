import pyrealsense2 as rs
pipeline = rs.pipeline()
pipeline.start()
print("RealSense 카메라가 정상적으로 동작합니다!")
pipeline.stop()