import numpy as np
import numpy.linalg as LA
import cv2
import uuid
from unique_identifier_msgs.msg import UUID


class ObstacleClass:
    """wrap a kalman filter and extra information for one single obstacle

    State space is 3D (x, y, z) by default, if you want to work on 2D (for example top-down view), you can simply make z a constant value and independent of x, y.

    Arrtibutes:
        position: 3d position of center point, numpy array with shape (3, 1)
        velocity: 3d velocity of center point, numpy array with shape (3, 1)
        kalman: cv2.KalmanFilter
        dying: count missing frames for this obstacle, if reach threshold, delete this obstacle
    """

    def __init__(
        self,
        obstacle_msg,
        top_down,
        measurement_noise_cov,
        error_cov_post,
        process_noise_cov,
    ):
        """Initialize with an Obstacle msg and an assigned id"""
        self.msg = obstacle_msg

        uuid_msg = UUID()
        uuid_msg.uuid = list(uuid.uuid4().bytes)
        self.msg.uuid = uuid_msg

        position = np.array(
            [
                [
                    obstacle_msg.position.x,
                    obstacle_msg.position.y,
                    obstacle_msg.position.z,
                ]
            ]
        ).T  # shape 3*1
        velocity = np.array(
            [
                [
                    obstacle_msg.velocity.x,
                    obstacle_msg.velocity.y,
                    obstacle_msg.velocity.z,
                ]
            ]
        ).T

        # check aganist state space dimension, top_down or not
        if top_down:
            measurement_noise_cov[2] = 0.0
            error_cov_post[2] = 0.0
            error_cov_post[5] = 0.0
            process_noise_cov[2] = 0.0

        # setup kalman filter
        self.kalman = cv2.KalmanFilter(
            6, 3
        )  # 3d by default, 6d state space and 3d observation space
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], np.float32
        )
        self.kalman.measurementNoiseCov = np.diag(measurement_noise_cov).astype(
            np.float32
        )
        self.kalman.statePost = np.concatenate([position, velocity]).astype(np.float32)
        self.kalman.errorCovPost = np.diag(error_cov_post).astype(np.float32)

        self.dying = 0
        self.top_down = top_down
        self.process_noise_cov = process_noise_cov
        
        ## 공분산 반영 하도록 수정 ##
        self.base_R = np.diag(measurement_noise_cov).astype(np.float32)  # 기본 R 저장
        self.kalman.measurementNoiseCov = self.base_R.copy()

        self.last_meas_pos = position.astype(np.float32)          # 3x1
        self.last_meas_cov = np.eye(3, dtype=np.float32) * 0.05   # 초기 게이트용(작게)

    def predict(self, dt):
        """update F and Q matrices, call KalmanFilter.predict and store position and velocity"""

        # construct new transition matrix
        """
        F = 1, 0, 0, dt, 0,  0
            0, 1, 0, 0,  dt, 0
            0, 0, 1, 0,  0,  dt
            0, 0, 0, 1,  0,  0
            0, 0, 0, 0,  1,  0
            0, 0, 0, 0,  0,  1
        """
        F = np.eye(6).astype(np.float32)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # construct new process conv matrix
        """assume constant velocity, and obstacle's acceleration has noise in x, y, z direction ax, ay, az
        Q = dt4*ax/4, 0,        0,        dt3*ax/2, 0,        0
            0,        dt4*ay/4, 0,        0,        dt3*ay/2, 0
            0,        0,        dt4*az/4, 0,        0,        dt3*az/2
            dt3*ax/2, 0,        0,        dt2*ax,   0,        0
            0,        dt3*ay/2, 0,        0,        dt2*ay,   0
            0,        0,        dt3*az/2, 0,        0,        dt2*az
        """
        dt2 = dt**2
        dt3 = dt * dt2
        dt4 = dt2**2

        Q = np.array([[dt4*self.process_noise_cov[0]/4, 0, 0, dt3*self.process_noise_cov[0]/2, 0, 0],
                      [0, dt4*self.process_noise_cov[1]/4, 0, 0, dt3*self.process_noise_cov[1]/2, 0],
                      [0, 0, dt4*self.process_noise_cov[2]/4, 0, 0, dt3*self.process_noise_cov[2]/2],
                      [dt3*self.process_noise_cov[0]/2, 0, 0, dt2*self.process_noise_cov[0], 0, 0],
                      [0, dt3*self.process_noise_cov[1]/2, 0, 0, dt2*self.process_noise_cov[1], 0],
                      [0, 0, dt3*self.process_noise_cov[2]/2, 0, 0, dt2*self.process_noise_cov[2]]]).astype(np.float32)

        self.kalman.transitionMatrix = F
        self.kalman.processNoiseCov = Q
        self.kalman.predict()
        self.msg.position.x = float(self.kalman.statePre[0][0])
        self.msg.position.y = float(self.kalman.statePre[1][0])
        self.msg.position.z = float(self.kalman.statePre[2][0])
        self.msg.velocity.x = float(self.kalman.statePre[3][0])
        self.msg.velocity.y = float(self.kalman.statePre[4][0])
        self.msg.velocity.z = float(self.kalman.statePre[5][0])

    # def correct(self, detect_msg):
    #     """extract position as measurement and update KalmanFilter"""
    #     if self.top_down:
    #         detect_msg.position.z = 0.0
    #     measurement = np.array(
    #         [[detect_msg.position.x, detect_msg.position.y, detect_msg.position.z]]
    #     ).T.astype(np.float32)
    #     self.kalman.correct(measurement)
    #     self.msg.position.x = float(self.kalman.statePost[0][0])
    #     self.msg.position.y = float(self.kalman.statePost[1][0])
    #     self.msg.position.z = float(self.kalman.statePost[2][0])
    #     self.msg.velocity.x = float(self.kalman.statePost[3][0])
    #     self.msg.velocity.y = float(self.kalman.statePost[4][0])
    #     self.msg.velocity.z = float(self.kalman.statePost[5][0])
    #     self.msg.size = detect_msg.size
    #     self.msg.position_covariance = detect_msg.position_covariance
    #     self.msg.velocity_covariance = detect_msg.velocity_covariance

    # def distance(self, other_msg):
    #     """measurement distance between two obstacles, dy default it's Euler distance between centers
    #     you can extent the Obstacle msg to include more features like class or uncertainty and include in the distance function"""
    #     position = np.array(
    #         [[self.msg.position.x, self.msg.position.y, self.msg.position.z]]
    #     ).T
    #     other_position = np.array(
    #         [[other_msg.position.x, other_msg.position.y, other_msg.position.z]]
    #     ).T
    #     return np.linalg.norm(position - other_position)
    
    
    ## 속도, 위치 공분산을 포함한 함수 ##
    def correct(self, detect_msg):
        """extract position as measurement and update KalmanFilter"""
        if self.top_down:
            detect_msg.position.z = 0.0

        # --- (1) R(측정 공분산) 위생 처리 ---
        R = self.base_R.copy()
        if len(detect_msg.position_covariance) == 9:
            Rm = np.array(detect_msg.position_covariance, dtype=np.float32).reshape(3,3)
            if np.isfinite(Rm).all():
                # 대칭화 + 대각 최소치/최대치 보장
                Rm = 0.5*(Rm + Rm.T)
                d = np.clip(np.diag(Rm), 1e-6, 1e3)
                Rm = np.diag(d).astype(np.float32)
                R = Rm
        self.kalman.measurementNoiseCov = R

        # --- (2) 측정 벡터 구성 ---
        measurement = np.array(
            [[detect_msg.position.x, detect_msg.position.y, detect_msg.position.z]],
            dtype=np.float32
        ).T

        # --- (3) correct() 호출 전 백업 (롤백용) ---
        xpost_prev = self.kalman.statePost.copy()
        Ppost_prev = self.kalman.errorCovPost.copy()

        # --- (4) 칼만 업데이트 ---
        try:
            self.kalman.correct(measurement)
        except Exception:
            # OpenCV 내부 수치문제 방지: 롤백하고 해당 프레임 업데이트 스킵
            self.kalman.statePost = xpost_prev
            self.kalman.errorCovPost = Ppost_prev
            self.dying += 1
            return

        # --- (5) correct() 직후 산출값 검사 + 롤백/보정 ---
        xpost = self.kalman.statePost.astype(np.float32)
        Ppost = self.kalman.errorCovPost.astype(np.float32)

        if (not np.isfinite(xpost).all()) or (not np.isfinite(Ppost).all()):
            # 롤백 + 프레임 미관측 취급
            self.kalman.statePost = xpost_prev
            self.kalman.errorCovPost = Ppost_prev
            self.dying += 1
            return

        # 공분산 대칭화 + 대각 하한/상한 클립 (PSD 보장 수준으로 안전하게)
        Ppost = 0.5 * (Ppost + Ppost.T)
        diag = np.clip(np.diag(Ppost), 1e-6, 1e6).astype(np.float32)
        Ppost = np.diag(diag)
        self.kalman.errorCovPost = Ppost

        # --- (6) 상태/공분산을 메시지에 반영 ---
        self.msg.position.x = float(self.kalman.statePost[0][0])
        self.msg.position.y = float(self.kalman.statePost[1][0])
        self.msg.position.z = float(self.kalman.statePost[2][0])
        self.msg.velocity.x = float(self.kalman.statePost[3][0])
        self.msg.velocity.y = float(self.kalman.statePost[4][0])
        self.msg.velocity.z = float(self.kalman.statePost[5][0])
        self.msg.size = detect_msg.size

        Ppost = self.kalman.errorCovPost.astype(np.float32)
        self.msg.position_covariance = Ppost[0:3, 0:3].reshape(-1).tolist()
        self.msg.velocity_covariance = Ppost[3:6, 3:6].reshape(-1).tolist()

        # 마지막 관측(원시 측정)도 저장 (게이팅용)
        self.last_meas_pos = np.array(
            [[detect_msg.position.x, detect_msg.position.y, detect_msg.position.z]],
            dtype=np.float32
        ).T
        self.last_meas_cov = Ppost[0:3, 0:3] + np.eye(3, dtype=np.float32) * 1e-6

      

    ## 속도, 위치 공분산을 포함한 함수 ##
    # def distance(self, other_msg):
    #     mu = np.array([[self.msg.position.x, self.msg.position.y, self.msg.position.z]],
    #                   dtype=np.float32).T
    #     z  = np.array([[other_msg.position.x, other_msg.position.y, other_msg.position.z]],
    #                   dtype=np.float32).T
    #     if not np.isfinite(mu).all() or not np.isfinite(z).all():
    #         return float('inf')

    #     P = self.kalman.errorCovPre.astype(np.float32)
    #     Ppos = P[0:3, 0:3]
    #     if not np.isfinite(Ppos).all():
    #         return float('inf')

    #     R = np.eye(3, dtype=np.float32)
    #     if len(other_msg.position_covariance) == 9:
    #         Rm = np.array(other_msg.position_covariance, dtype=np.float32).reshape(3,3)
    #         if np.isfinite(Rm).all():
    #             R = Rm
    #     R += np.eye(3, dtype=np.float32) * 1e-6  # 수치 안정화

    #     S = Ppos + R
    #     diff = z - mu

    #     # ▶ 역행렬 대신 촐레스키/선형해결 사용
    #     try:
    #         L = np.linalg.cholesky(S)           # S = L L^T
    #         y = np.linalg.solve(L, diff)        # L y = diff
    #         y = np.linalg.solve(L.T, y)         # L^T y = previous y (S^{-1} diff)
    #         d2 = float(diff.T @ y)              # Mahalanobis^2
    #     except np.linalg.LinAlgError:
    #         try:
    #             y = np.linalg.solve(S, diff)    # 일반 해법 (여전히 inv보다 빠름)
    #             d2 = float(diff.T @ y)
    #         except np.linalg.LinAlgError:
    #             return float('inf')

    #     if not np.isfinite(d2) or d2 < 0.0:
    #         return float('inf')
    #     return d2**0.5

    def distance(self, other_msg):
        # 측정 z
        z  = np.array([[other_msg.position.x, other_msg.position.y, other_msg.position.z]], dtype=np.float32).T
        if not np.isfinite(z).all():
            return float('inf')
    
        # ▶ 관측이 끊긴 동안(dying>0)에는 "마지막 관측"을 기준으로 연결(표시는 안 함)
        if self.dying > 0:
            mu   = self.last_meas_pos.astype(np.float32)                 # last measurement mean
            Ppos = self.last_meas_cov.astype(np.float32)                 # last measurement covariance (게이트 타이트)
        else:
            mu = np.array([[self.msg.position.x, self.msg.position.y, self.msg.position.z]], dtype=np.float32).T
            P   = self.kalman.errorCovPre.astype(np.float32)
            Ppos= P[0:3, 0:3]
    
        if not np.isfinite(mu).all() or not np.isfinite(Ppos).all():
            return float('inf')
    
        R = np.eye(3, dtype=np.float32)
        if len(other_msg.position_covariance) == 9:
            Rm = np.array(other_msg.position_covariance, dtype=np.float32).reshape(3,3)
            if np.isfinite(Rm).all():
                R = Rm
        R += np.eye(3, dtype=np.float32) * 1e-6
    
        S = Ppos + R
        diff = z - mu
    
        try:
            L = np.linalg.cholesky(S)
            y = np.linalg.solve(L, diff)
            y = np.linalg.solve(L.T, y)
            d2 = float(diff.T @ y)
        except np.linalg.LinAlgError:
            try:
                y = np.linalg.solve(S, diff)
                d2 = float(diff.T @ y)
            except np.linalg.LinAlgError:
                return float('inf')
    
        if not np.isfinite(d2) or d2 < 0.0:
            return float('inf')
        return d2**0.5