import uuid, colorsys, numpy as np

import math
from scipy.optimize import linear_sum_assignment

from nav2_dynamic_msgs.msg import Obstacle, ObstacleArray
from visualization_msgs.msg import Marker, MarkerArray

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import copy

from kf_hungarian_tracker.obstacle_class import ObstacleClass

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_point, do_transform_vector3
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.qos import qos_profile_sensor_data

from builtin_interfaces.msg import Duration



class KFHungarianTracker(Node):
    """Use Kalman Fiter and Hungarian algorithm to track multiple dynamic obstacles

    Use Hungarian algorithm to match presenting obstacles with new detection and maintain a kalman filter for each obstacle.
    spawn ObstacleClass when new obstacles come and delete when they disappear for certain number of frames

    Attributes:
        obstacle_list: a list of ObstacleClass that currently present in the scene
        sec, nanosec: timing from sensor msg
        detection_sub: subscrib detection result from detection node
        tracker_obstacle_pub: publish tracking obstacles with ObstacleArray
        tracker_pose_pub: publish tracking obstacles with PoseArray, for rviz visualization
    """

    def __init__(self):
        """initialize attributes and setup subscriber and publisher"""

        super().__init__("kf_hungarian_node")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("global_frame", "map"),
                ("process_noise_cov", [2.0, 2.0, 0.5]),
                ("top_down", False),
                ("death_threshold", 3),
                ("measurement_noise_cov", [1.0, 1.0, 1.0]),
                ("error_cov_post", [1.0, 1.0, 1.0, 10.0, 10.0, 10.0]),
                ("vel_filter", [0.1, 2.0]),
                ("height_filter", [-2.0, 2.0]),
                ("cost_filter", 1.0),
                ("transform_to_global_frame", True),
                ("infer_orientation_from_velocity", True),          # True
            ],
        )
        self.global_frame = self.get_parameter("global_frame")._value
        self.death_threshold = self.get_parameter("death_threshold")._value
        self.measurement_noise_cov = self.get_parameter("measurement_noise_cov")._value
        self.error_cov_post = self.get_parameter("error_cov_post")._value
        self.process_noise_cov = self.get_parameter("process_noise_cov")._value
        self.vel_filter = self.get_parameter("vel_filter")._value
        self.height_filter = self.get_parameter("height_filter")._value
        self.top_down = self.get_parameter("top_down")._value
        self.cost_filter = self.get_parameter("cost_filter")._value
        self.transform_to_global_frame = self.get_parameter(
            "transform_to_global_frame"
        )._value
        self.infer_orientation_from_velocity = self.get_parameter(
            "infer_orientation_from_velocity"
        )._value

        self.obstacle_list = []
        self.sec = 0
        self.nanosec = 0

        # subscribe to detector  → ★ BestEffort
        self.detection_sub = self.create_subscription(
            ObstacleArray, "/aeirobot/vslam_dynamic_obstacles", self.callback, qos_profile_sensor_data
        )

        # publisher for tracking result → ★ BestEffort
        self.tracker_obstacle_pub = self.create_publisher(ObstacleArray, "tracking", qos_profile_sensor_data)
        self.tracker_marker_pub = self.create_publisher(
            MarkerArray, "tracking_marker", qos_profile_sensor_data
        )

        # setup tf related (그대로)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


    def callback(self, msg):
        """callback function for detection result"""

        # update delta time
        dt = (msg.header.stamp.sec - self.sec) + (
            msg.header.stamp.nanosec - self.nanosec
        ) / 1e9
        self.sec = msg.header.stamp.sec
        self.nanosec = msg.header.stamp.nanosec

        # ← 비정상 dt 방어
        if not np.isfinite(dt):
            dt = 0.0
        dt = max(0.0, min(dt, 1.0))

        # get detection
        detections = msg.obstacles

        # kalman predict
        for obj in self.obstacle_list:
            obj.predict(dt)

        if (self.transform_to_global_frame) and (self.global_frame is not None):
            try:
                # ★ 카메라 프레임 시각
                stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        
                # 동일 프레임이면 패스
                if msg.header.frame_id != self.global_frame:
                    # ★ 해당 시각 TF가 준비됐는지 확인 (최대 0.2s 대기)
                    if not self.tf_buffer.can_transform(
                        self.global_frame, msg.header.frame_id, stamp, Duration(seconds=0.2)
                    ):
                        self.get_logger().warn(
                            f"TF {msg.header.frame_id}->{self.global_frame} @ {stamp.nanoseconds}ns 미준비. 프레임 스킵"
                        )
                        return
        
                    # ★ 정확히 그 시각의 TF로 변환
                    trans = self.tf_buffer.lookup_transform(
                        self.global_frame, msg.header.frame_id, stamp
                    )
        
                    msg.header.frame_id = self.global_frame
        
                    translation_backup_x = trans.transform.translation.x
                    translation_backup_y = trans.transform.translation.y
                    translation_backup_z = trans.transform.translation.z
        
                    for i in range(len(detections)):
                        # 매 루프에서 translation이 변형될 수 있어 백업값으로 복원
                        trans.transform.translation.x = translation_backup_x
                        trans.transform.translation.y = translation_backup_y
                        trans.transform.translation.z = translation_backup_z
        
                        p = PointStamped()
                        p.point = detections[i].position
                        detections[i].position = do_transform_point(p, trans).point
        
                        v = Vector3Stamped()
                        v.vector = detections[i].velocity
                        detections[i].velocity = do_transform_vector3(v, trans).vector
        
                        s = Vector3Stamped()
                        s.vector = detections[i].size
                        detections[i].size = do_transform_vector3(s, trans).vector
                # 같으면 변환 불필요
            except TransformException as ex:
                self.get_logger().error(
                    f"fail to get tf from {msg.header.frame_id} to {self.global_frame}: {ex}"
                )
                return

        # 유한성 필터
        def _finite_obstacle(o):
            vals = [o.position.x, o.position.y, o.position.z,
                    o.velocity.x, o.velocity.y, o.velocity.z,
                    o.size.x, o.size.y, o.size.z]
            ok_basic = all(np.isfinite(v) for v in vals)
            if len(o.position_covariance) == 9:
                Rc = np.array(o.position_covariance, dtype=np.float32)
                ok_cov = np.isfinite(Rc).all()
            else:
                ok_cov = True
            return ok_basic and ok_cov
        detections = [d for d in detections if _finite_obstacle(d)]

        num_of_detect = len(detections)
        num_of_obstacle = len(self.obstacle_list)

        no_detection = False

        if num_of_detect == 0 and num_of_obstacle == 0:
            return

        if num_of_obstacle == 0:
            self.birth([], num_of_detect, detections)
            return

        if num_of_detect == 0:
            obs_ind, det_ind = [], []
            no_detection = True
        else:
            BIG = 1e6
            cost = np.full((num_of_obstacle, num_of_detect), BIG, dtype=np.float32)

            for i in range(num_of_obstacle):
                Ppos = self.obstacle_list[i].kalman.errorCovPre[:3, :3].astype(np.float32)
                if not np.isfinite(Ppos).all():
                    continue
                Pdiag = np.diag(Ppos)
                mu = np.array([self.obstacle_list[i].msg.position.x,
                               self.obstacle_list[i].msg.position.y,
                               self.obstacle_list[i].msg.position.z],
                              dtype=np.float32)
                for j in range(num_of_detect):
                    z = np.array([detections[j].position.x,
                                  detections[j].position.y,
                                  detections[j].position.z],
                                 dtype=np.float32)
                    if not np.isfinite(z).all():
                        continue
                    Rdiag = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                    if len(detections[j].position_covariance) == 9:
                        Rm = np.array(detections[j].position_covariance, dtype=np.float32).reshape(3,3)
                        if np.isfinite(Rm).all():
                            Rdiag = np.diag(Rm)
                    sigma = np.sqrt(np.clip(Pdiag + Rdiag, 1e-6, 1e9))
                    gate = float(self.cost_filter)
                    if np.any(np.abs(z - mu) > gate * sigma):
                        continue
                    d = self.obstacle_list[i].distance(detections[j])
                    if np.isfinite(d):
                        cost[i, j] = d

            gate = float(self.cost_filter) * 3.0
            cost[cost > gate] = BIG
            cost = np.nan_to_num(cost, nan=BIG, posinf=BIG, neginf=BIG)

            row_mask = ~(np.all(cost >= BIG, axis=1))
            col_mask = ~(np.all(cost >= BIG, axis=0))

            if not row_mask.any() or not col_mask.any():
                obs_ind, det_ind = [], []
            else:
                sub_cost = cost[np.ix_(row_mask, col_mask)]
                try:
                    sub_obs_ind, sub_det_ind = linear_sum_assignment(sub_cost)
                    obs_map = np.where(row_mask)[0]
                    det_map = np.where(col_mask)[0]
                    obs_ind = [int(obs_map[r]) for r in sub_obs_ind]
                    det_ind = [int(det_map[c]) for c in sub_det_ind]
                except ValueError as e:
                    self.get_logger().warn(f'Hungarian infeasible this frame: {e}. Skip matching.')
                    obs_ind, det_ind = [], []

            new_obs_ind, new_det_ind = [], []
            for o, d in zip(obs_ind, det_ind):
                if cost[o, d] < self.cost_filter:
                    new_obs_ind.append(o)
                    new_det_ind.append(d)
            obs_ind, det_ind = new_obs_ind, new_det_ind

        # kalman update
        for o, d in zip(obs_ind, det_ind):
            self.obstacle_list[o].correct(detections[d])

        # birth: 디텍션 있는 프레임에서만
        if not no_detection:
            self.birth(det_ind, num_of_detect, detections)

        # death: 항상 호출해서 dying 갱신
        dead_object_list = self.death(obs_ind, num_of_obstacle)

        # ====== 변경점: dying 기간에도 퍼블리시(임계 미만) → 깜빡임 방지 ======
        filtered_obstacle_list = []
        for obs in self.obstacle_list:
            if obs.dying >= self.death_threshold:
                continue  # 실제로 죽은 트랙만 제외
            obs_vel = np.linalg.norm([obs.msg.velocity.x, obs.msg.velocity.y, obs.msg.velocity.z])
            obs_height = obs.msg.position.z
            if (self.vel_filter[0] < obs_vel < self.vel_filter[1]
                and self.height_filter[0] < obs_height < self.height_filter[1]):
                filtered_obstacle_list.append(obs)
        # ===============================================================

        # 안전 퍼블리시
        def _sanitize_in_place(m):
            core = [m.position.x, m.position.y, m.position.z,
                    m.velocity.x, m.velocity.y, m.velocity.z,
                    m.size.x,    m.size.y,    m.size.z]
            if not all(np.isfinite(core)):
                return False
            if len(m.position_covariance) == 9:
                Rc = np.array(m.position_covariance, np.float32)
                if not np.isfinite(Rc).all():
                    Rc = np.eye(3, dtype=np.float32) * 0.05
                    m.position_covariance = Rc.reshape(-1).tolist()
                else:
                    Rc = Rc.reshape(3,3)
                    Rc = 0.5*(Rc + Rc.T)
                    d  = np.clip(np.diag(Rc), 1e-6, 1e6)
                    m.position_covariance = np.diag(d).astype(np.float32).reshape(-1).tolist()
            if len(m.velocity_covariance) == 9:
                Rv = np.array(m.velocity_covariance, np.float32)
                if not np.isfinite(Rv).all():
                    Rv = np.eye(3, dtype=np.float32) * 0.1
                    m.velocity_covariance = Rv.reshape(-1).tolist()
                else:
                    Rv = Rv.reshape(3,3)
                    Rv = 0.5*(Rv + Rv.T)
                    d  = np.clip(np.diag(Rv), 1e-6, 1e6)
                    m.velocity_covariance = np.diag(d).astype(np.float32).reshape(-1).tolist()
            return True

        safe_obs = []
        for obs in filtered_obstacle_list:
            if _sanitize_in_place(obs.msg):
                safe_obs.append(obs)
            else:
                self.get_logger().warn("Drop NaN/Inf track before publish")

        if self.tracker_obstacle_pub.get_subscription_count() > 0:
            obstacle_array = ObstacleArray()
            obstacle_array.header = msg.header
            obstacle_array.obstacles = [o.msg for o in safe_obs]
            self.tracker_obstacle_pub.publish(obstacle_array)

        # 기존 marker_array 생성/append/publish 블록 삭제하고 아래 한 줄로 교체
        self.publish_tracking_markers(safe_obs, msg.header, dead_object_list)

      

    def birth(self, det_ind, num_of_detect, detections):
        """generate new ObstacleClass for detections that do not match any in current obstacle list"""
        for det in range(num_of_detect):
            if det not in det_ind:
                obstacle = ObstacleClass(
                    detections[det],
                    self.top_down,
                    self.measurement_noise_cov,
                    self.error_cov_post,
                    self.process_noise_cov,
                )
                self.obstacle_list.append(obstacle)

    def death(self, obj_ind, num_of_obstacle):
        """count obstacles' missing frames and delete when reach threshold"""
        new_object_list = []
        dead_object_list = []
        # for previous obstacles
        for obs in range(num_of_obstacle):
            if obs not in obj_ind:
                self.obstacle_list[obs].dying += 1
            else:
                self.obstacle_list[obs].dying = 0

            if self.obstacle_list[obs].dying < self.death_threshold:
                new_object_list.append(self.obstacle_list[obs])
            else:
                obstacle_uuid = uuid.UUID(
                    bytes=bytes(self.obstacle_list[obs].msg.uuid.uuid)
                )
                dead_object_list.append(obstacle_uuid)

        # add newly born obstacles
        for obs in range(num_of_obstacle, len(self.obstacle_list)):
            new_object_list.append(self.obstacle_list[obs])

        self.obstacle_list = new_object_list
        return dead_object_list

    def publish_tracking_markers(self, safe_obs, header, dead_object_list):
        """RViz 마커를 안전하게 퍼블리시:
           - 프레임 시작에 DELETEALL로 싹 지움
           - 새 마커에는 lifetime을 짧게 줘서 유령 방지
        """
        marr = MarkerArray()

        # 0) 이전 것 전부 삭제 (매 프레임)
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        marr.markers.append(clear)

        # 1) 현재 트랙들 그리기
        for obs in safe_obs:
            m = obs.msg
            obstacle_uuid = uuid.UUID(bytes=bytes(m.uuid.uuid))
            (r, g, b) = colorsys.hsv_to_rgb(obstacle_uuid.int % 360 / 360.0, 1.0, 1.0)

            # 공통 lifetime (예: 0.4초) – 주기적으로 다시 퍼블리시되므로 끊기지 않음
            life = Duration(sec=0, nanosec=400_000_000)

            # cube
            cube = Marker()
            cube.header = header
            cube.ns = str(obstacle_uuid)
            cube.id = 0
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.color.a = 0.5; cube.color.r = r; cube.color.g = g; cube.color.b = b
            cube.pose.position = m.position
            angle = float(np.arctan2(m.velocity.y, m.velocity.x)) if np.isfinite([m.velocity.x, m.velocity.y]).all() else 0.0
            if self.infer_orientation_from_velocity:
                cube.pose.orientation.z = float(np.sin(angle / 2))
                cube.pose.orientation.w = float(np.cos(angle / 2))
            else:
                cube.pose.orientation.z = 0.0
                cube.pose.orientation.w = 1.0
            cube.scale = m.size
            cube.lifetime = life
            marr.markers.append(cube)

            # arrow
            arrow = Marker()
            arrow.header = header
            arrow.ns = str(obstacle_uuid)
            arrow.id = 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.color.a = 1.0; arrow.color.r = r; arrow.color.g = g; arrow.color.b = b
            arrow.pose.position = m.position
            arrow.pose.orientation.z = float(np.sin(angle / 2))
            arrow.pose.orientation.w = float(np.cos(angle / 2))
            vel_norm = float(np.linalg.norm([m.velocity.x, m.velocity.y, m.velocity.z]))
            if not np.isfinite(vel_norm):
                vel_norm = 0.0
            arrow.scale.x = vel_norm; arrow.scale.y = 0.05; arrow.scale.z = 0.05
            arrow.lifetime = life
            marr.markers.append(arrow)

        # 2) 죽은 애들 삭제 마커(DELETE)는 남겨두되, 매 프레임 DELETEALL을 먼저 보내므로
        #    누락되더라도 다음 프레임에 깨끗해집니다.
        for dead_uuid in dead_object_list:
            del_cube = Marker()
            del_cube.header = header
            del_cube.ns = str(dead_uuid); del_cube.id = 0
            del_cube.action = Marker.DELETE
            marr.markers.append(del_cube)

            del_arrow = Marker()
            del_arrow.header = header
            del_arrow.ns = str(dead_uuid); del_arrow.id = 1
            del_arrow.action = Marker.DELETE
            marr.markers.append(del_arrow)

        self.tracker_marker_pub.publish(marr)


def main(args=None):
    rclpy.init(args=args)

    node = KFHungarianTracker()
    node.get_logger().info("start spining tracker node...")

    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
