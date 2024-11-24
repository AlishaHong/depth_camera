import numpy as np
import open3d as o3d
import pyrealsense2 as rs

# 1. RealSense 파이프라인 초기화
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

try:
    while True:
        # 2. 프레임 수집
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # 3. 뎁스 이미지를 NumPy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())

        # 4. 카메라 내보정 값 (Intrinsic Parameters)
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy

        # 5. 3D 포인트 클라우드 생성
        points = []
        for v in range(depth_image.shape[0]):
            for u in range(depth_image.shape[1]):
                z = depth_image[v, u] * 0.001  # 깊이 값을 미터 단위로 변환
                if z == 0:  # 깊이 값이 0인 경우 건너뜀
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append((x, y, z))

        # 6. Open3D를 사용하여 포인트 클라우드 시각화
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
        o3d.visualization.draw_geometries([point_cloud])  # 3D 시각화 창 열기

        break  # 한 프레임만 시각화 후 종료 (필요에 따라 수정 가능)
finally:
    pipeline.stop()