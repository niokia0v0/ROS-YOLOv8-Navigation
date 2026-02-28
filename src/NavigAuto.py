#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from ultralytics import YOLO

class NavigAuto:
    def __init__(self):
        rospy.init_node('detecteur_navig_node')
        self.bridge = CvBridge()
        self.IMG_SIZE = 128 
        self.IMG_CENTER = self.IMG_SIZE // 2

        # Memory variable
        self.last_road_target_X = self.IMG_CENTER 

        # Load Model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "best.onnx")
        self.model = YOLO(model_path, task='segment')

        self.class_names = {
            0: 'flash1', 1: 'flash2', 2: 'inverse1', 3: 'inverse2',
            4: 'line', 5: 'parking', 6: 'pieton', 7: 'road', 8: 'robot'
        }
        self.PIETON_ID = 6
        self.ROAD_ID = 7
        self.ROBOT_ID = 8 
        self.PARKING_ID = 5
        self.FLASH2_ID = 1
        
        self.subscriber = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.camera_callback, queue_size=1)
        self.pub_annotated = rospy.Publisher('/detecteur_route/image_annotated/compressed', CompressedImage, queue_size=1)
        self.pub_data = rospy.Publisher('/navig_auto/data', Point, queue_size=1)

    def camera_callback(self, msg):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_AREA)
            
            # Confidence threshold set to 0.25 (Parking handled separately below)
            results = self.model.predict(frame, imgsz=self.IMG_SIZE, verbose=False, conf=0.25, device='cpu')
            result = results[0]

            m_road = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.uint8)
            m_line = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.uint8)
            m_robot = np.zeros((self.IMG_SIZE, self.IMG_SIZE), dtype=np.uint8)
            vis_frame = frame.copy()
            
            detected_pieton_contours = []
            detected_parking_contours = []
            max_conf_inverse = 0.0
            max_conf_flash1 = 0.0
            max_conf_flash2 = 0.0 

            # --- Real-time Log List ---
            log_detections = []

            if result.masks is not None:
                for i, mask_data in enumerate(result.masks.data):
                    class_id = int(result.boxes.cls[i])
                    conf = float(result.boxes.conf[i])
                    class_name = self.class_names.get(class_id, str(class_id))
                    
                    # Collect detection info for printing
                    log_detections.append(f"{class_name}:{conf:.2f}")

                    if class_id == 0: max_conf_flash1 = max(max_conf_flash1, conf)
                    elif class_id == 1: max_conf_flash2 = max(max_conf_flash2, conf)
                    elif class_id in [2, 3]: max_conf_inverse = max(max_conf_inverse, conf)

                    mask_np = (mask_data.cpu().numpy() * 255).astype(np.uint8)
                    mask_np = cv2.resize(mask_np, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if class_id == self.ROAD_ID:
                        cv2.drawContours(m_road, contours, -1, 255, -1)
                    elif class_id == 4: # Line
                        cv2.drawContours(m_line, contours, -1, 255, -1)
                    elif class_id == self.PIETON_ID:
                        if conf > 0.25:
                            for c in contours: detected_pieton_contours.append(c)
                    elif class_id == self.ROBOT_ID: 
                        if conf > 0.25:
                            cv2.drawContours(m_robot, contours, -1, 255, -1)
                    elif class_id == self.PARKING_ID:
                        # Parking Threshold 0.3
                        if conf > 0.30:
                            for c in contours: detected_parking_contours.append(c)

            # --- Print Detections to Terminal ---
            if log_detections:
                print(f"[Detect] {', '.join(log_detections)}")

            # ================= Logic Decision & Priority Mux =================
            status_flag = 0.0 
            payload_z = -1000.0 

            # --- 1. Robot Check (Priority: High) ---
            is_robot_wait = False
            rows_with_robot = np.count_nonzero(np.sum(m_robot, axis=1))
            if rows_with_robot > (self.IMG_SIZE * 0.60):
                is_robot_wait = True
                cv2.putText(vis_frame, "ROBOT!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- 2. Blocked/Wall Check (Priority: High) ---
            is_blocked = False
            crash_start_row = int(self.IMG_SIZE * 0.68)
            crash_zone = m_line[crash_start_row:, :]
            cols_with_line = np.count_nonzero(np.sum(crash_zone, axis=0))
            if cols_with_line > (self.IMG_SIZE * 0.8):
                is_blocked = True
                cv2.putText(vis_frame, "BLOCKED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- 3. Parking Logic (Priority: Info) ---
            is_parking_seen = False
            is_parking_touch = False
            parking_err = -1000.0
            
            if detected_parking_contours:
                c_park = max(detected_parking_contours, key=cv2.contourArea)
                M_pk = cv2.moments(c_park)
                if M_pk["m00"] != 0:
                    cx_pk = int(M_pk["m10"] / M_pk["m00"])
                    parking_err = float(cx_pk - self.IMG_CENTER)
                    is_parking_seen = True
                    cv2.drawContours(vis_frame, [c_park], -1, (255, 0, 255), 2)
                    
                    _, y_pk, _, h_pk = cv2.boundingRect(c_park)
                    if (y_pk + h_pk) > (self.IMG_SIZE - 10):
                        is_parking_touch = True
                        cv2.putText(vis_frame, "TOUCH!", (cx_pk, y_pk), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

            # --- 4. Pieton Logic (Standard Info) ---
            if detected_pieton_contours:
                valid_candidates = []
                for c in detected_pieton_contours:
                    x, y, w, h = cv2.boundingRect(c)
                    if (y + h) > 115 and x < 95:
                        valid_candidates.append(c)
                if valid_candidates:
                    target_c = max(valid_candidates, key=lambda c: cv2.boundingRect(c)[1])
                    M_p = cv2.moments(target_c)
                    if M_p["m00"] != 0:
                        cx_p = int(M_p["m10"] / M_p["m00"])
                        payload_z = float(cx_p - self.IMG_CENTER)
                        cv2.drawContours(vis_frame, [target_c], -1, (0, 255, 0), 2)

            # --- 5. Road & Repulsion Logic ---
            horizon_row = 70
            roi_width = 30
            roi_mask = np.zeros_like(m_road)
            roi_mask[:, self.IMG_CENTER - roi_width : self.IMG_CENTER + roi_width] = 255
            road_slice = cv2.bitwise_and(m_road[horizon_row, :], roi_mask[horizon_row, :])
            road_pixels = np.where(road_slice > 0)[0]
            road_target_X = self.last_road_target_X
            if len(road_pixels) > 0:
                road_target_X = int((np.min(road_pixels) + np.max(road_pixels)) / 2.0)
                self.last_road_target_X = road_target_X

            err_repulsion = 0
            bottom_zone = m_line[80:, :]
            # Repulsion active even during parking to prevent line crossing
            if not is_blocked and np.sum(bottom_zone) > 0:
                M_l = cv2.moments(bottom_zone)
                if M_l["m00"] != 0:
                    cx_l = int(M_l["m10"] / M_l["m00"])
                    cy_l = int(M_l["m01"] / M_l["m00"])
                    dist_from_center = cx_l - self.IMG_CENTER
                    SAFE_MARGIN = 64 
                    REPULSION_GAIN = 6.5
                    if abs(dist_from_center) < SAFE_MARGIN:
                        penetration = SAFE_MARGIN - abs(dist_from_center)
                        direction_sign = np.sign(dist_from_center) if dist_from_center != 0 else 1
                        y_weight = (cy_l / self.IMG_SIZE) 
                        force = -1.0 * direction_sign * penetration * REPULSION_GAIN * (1.0 + y_weight)
                        err_repulsion = np.clip(force, -80, 80)
                        cv2.line(vis_frame, (self.IMG_CENTER, 0), (self.IMG_CENTER, 128), (0,0,255), 1)

            # ================= Status Output =================
            if is_robot_wait:
                status_flag = 3.0
            elif is_blocked:
                status_flag = 2.0
            elif (max_conf_inverse > 0.65 and max_conf_inverse > max_conf_flash1):
                status_flag = 1.0
            elif is_parking_touch:
                status_flag = 7.0
                payload_z = parking_err 
            elif is_parking_seen:
                status_flag = 6.0
                payload_z = parking_err 
            elif max_conf_flash2 > 0.8:
                status_flag = 5.0
            else:
                status_flag = 0.0

            # msg.x always contains repulsion
            data_msg = Point()
            data_msg.x = float((road_target_X - self.IMG_CENTER) + err_repulsion)
            data_msg.y = float(status_flag)
            data_msg.z = float(payload_z)
            self.pub_data.publish(data_msg)
            
            cv2.circle(vis_frame, (road_target_X, horizon_row), 3, (255, 0, 0), -1)
            msg_out = self.bridge.cv2_to_compressed_imgmsg(vis_frame)
            self.pub_annotated.publish(msg_out)

        except Exception as e:
            rospy.logerr(f"NavigAuto Error: {e}")

if __name__ == '__main__':
    NavigAuto()
    rospy.spin()