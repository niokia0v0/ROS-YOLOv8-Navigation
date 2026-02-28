#!/usr/bin/env python3
import rospy
import random
from geometry_msgs.msg import Twist, Point

# Etats
STATE_CRUISE = 0        
STATE_U_TURN_180 = 1    
STATE_ALIGN_PIETON = 2 
STATE_BLOCKED = 3       
STATE_WAIT_ROBOT = 4

STATE_INNER_ENTRY_TURN = 5   # Turn 90deg to inner loop
STATE_PARKING_GRAVITY = 6    # Parking gravity mode
STATE_PARKING_FINAL_MOVE = 7 # Parking final move (3s)
STATE_PARKING_WAIT = 8       # Parking wait
STATE_PARKING_EXIT = 9       # Parking exit (160deg)
STATE_SEARCHING_PARKING = 10 

Dice = 0.75

class ControleurNavig:
    def __init__(self):
        rospy.init_node('navigation_auto_node')
        self.rate = rospy.Rate(10)

        # --- PID & Speed ---
        self.kp_rot = 0.008      
        self.kp_align = 0.012 
        self.kp_parking = 0.005  
        self.linear_speed = 0.1
       
        
        # --- Action Parameters ---
        self.uturn_angular_speed = -1.4  
        self.uturn_duration = rospy.Duration(1.6)
        
        self.block_turn_speed = -1.4 
        self.block_turn_duration = rospy.Duration(0.8) 

        self.pieton_align_duration = rospy.Duration(2.0)
        self.pieton_cooldown_duration = rospy.Duration(5.0) 
        
        self.inner_turn_speed = -1.4 
        self.inner_turn_duration = rospy.Duration(0.8) 
        
        self.parking_exit_speed = -1.4 
        self.parking_exit_duration = rospy.Duration(1.4)

        self.parking_search_duration = rospy.Duration(22.0) 
        self.gravity_max_duration = rospy.Duration(15.0)     
        
        self.search_end_time = rospy.Time(0)

        # --- Mission Logic Variables ---
        self.mission_active = False       
        self.allow_parking_logic = False  
        self.pieton_pass_count = 0        
        
        # Consecutive Frame Counters
        self.flash2_counter = 0
        self.parking_counter = 0
        self.inverse_counter = 0
        self.INVERSE_TRIGGER_COUNT = 8
        
        # Quit parking
        self.abort_counter = 0
        self.ABORT_TRIGGER_COUNT = 10

        self.current_state = STATE_CRUISE
        self.previous_state = STATE_CRUISE 
        
        self.state_start_time = rospy.Time.now()
        self.pieton_cooldown_until = rospy.Time(0)
        self.mission_cooldown_until = rospy.Time(0) 
        self.is_inverse_confirmed = False

        self.road_error = 0.0 
        self.status_flag = 0.0 
        self.payload_z = -1000.0
        self.last_msg_time = rospy.Time.now()

        self.pieton_error = -1000.0
        self.parking_error = -1000.0

        self.pub_vel = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.sub_data = rospy.Subscriber('/navig_auto/data', Point, self.data_callback)

        rospy.loginfo("ControleurNavig initialized with High Priority Timeout & Robust Fail-safe.")

    def data_callback(self, msg):
        self.last_msg_time = rospy.Time.now()
        self.road_error = msg.x 
        self.status_flag = msg.y 
        self.payload_z = msg.z
        
        if abs(msg.y - 1.0) < 0.1: 
            self.inverse_counter += 1
        else: 
            self.inverse_counter = 0    
        self.is_inverse_confirmed = (self.inverse_counter >= self.INVERSE_TRIGGER_COUNT)

        self.pieton_error = -1000.0
        self.parking_error = -1000.0

        # Parking logic has priority on payload_z if status is 6 or 7
        if abs(msg.y - 6.0) < 0.1 or abs(msg.y - 7.0) < 0.1:
            if self.allow_parking_logic:
                self.parking_error = msg.z
        else:
            self.pieton_error = msg.z

    def run(self):
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if (now - self.last_msg_time).to_sec() > 1.0:
                self.stop_robot()
                self.rate.sleep()
                continue

            # ================= Priority Safety Checks =================
            # Checks that apply ONLY when actively searching for parking
            # These are placed here to ensure they are not blocked by other triggers
            if self.current_state == STATE_SEARCHING_PARKING:
                
                # A. Timeout Check
                if now > self.search_end_time:
                    rospy.logwarn("Parking Search TIMEOUT (22s). Aborting Mission. Returning to Cruise.")
                    self.reset_mission_vars()
                    self.change_state(STATE_CRUISE)
                    self.rate.sleep()
                    continue # Skip remainder of loop to apply Cruise immediately

                # B. Fail-safe: Abort if Signs (Flash2/Inverse) are seen continuously
                # 5.0 = Flash2, 1.0 = Inverse. 5 consecutive frames.
                if abs(self.status_flag - 5.0) < 0.1 or abs(self.status_flag - 1.0) < 0.1:
                    self.abort_counter += 1
                else:
                    self.abort_counter = 0
                
                if self.abort_counter >= self.ABORT_TRIGGER_COUNT:
                    rospy.logwarn(f"Abort Signal Confirmed ({self.ABORT_TRIGGER_COUNT} frames). Resetting to Cruise.")
                    self.abort_counter = 0 # Reset counter
                    self.reset_mission_vars()
                    self.change_state(STATE_CRUISE)
                    self.rate.sleep()
                    continue

            # ================= Trigger Logic =================
            # Flash2 Detection
            if abs(self.status_flag - 5.0) < 0.1:
                if not self.mission_active and now > self.mission_cooldown_until:
                    self.flash2_counter += 1
            else:
                self.flash2_counter = 0
            
            if self.flash2_counter >= 10 and not self.mission_active:
                dice = random.random()
                rospy.loginfo(f"Rolling Dice: {dice:.2f}")
                self.flash2_counter = 0 
                if dice < Dice:
                    rospy.loginfo(">>> MISSION START: Inner Loop Mode Activated <<<")
                    self.mission_active = True
                    self.pieton_pass_count = 0
                    self.allow_parking_logic = False 
                else:
                    rospy.loginfo("Mission Skipped.")
                    self.mission_active = False 

            # Parking Counter
            if self.allow_parking_logic and abs(self.status_flag - 6.0) < 0.1:
                self.parking_counter += 1
            else:
                self.parking_counter = 0

            # ================= State Transitions =================
            
            # 1. Robot Detection (Priority: Suspend & Resume)
            if abs(self.status_flag - 3.0) < 0.1:
                if self.current_state != STATE_WAIT_ROBOT:
                    rospy.loginfo(f"ROBOT!! Stopping. Saving state: {self.current_state}")
                    self.previous_state = self.current_state 
                    self.current_state = STATE_WAIT_ROBOT
            
            # If Robot is GONE and we were waiting, Resume previous state
            elif self.current_state == STATE_WAIT_ROBOT:
                rospy.loginfo(f"ROBOT GONE. Resuming state: {self.previous_state}")
                self.current_state = self.previous_state
                # Note: We do not reset state_start_time here to allow timeouts to continue where left off

            # 2. Block Detection (Priority: Avoid & Resume)
            # Ignored during critical maneuvers (Parking Gravity/Final, U-Turn, Inner Turn)
            elif abs(self.status_flag - 2.0) < 0.1:
                if self.current_state not in [STATE_BLOCKED, STATE_WAIT_ROBOT, STATE_PARKING_GRAVITY, 
                                              STATE_PARKING_FINAL_MOVE, STATE_U_TURN_180, STATE_INNER_ENTRY_TURN]:
                    rospy.logwarn(f"WALL BLOCKED! Emergency Turn. Saving state: {self.current_state}")
                    self.previous_state = self.current_state 
                    self.change_state(STATE_BLOCKED)

            # 3. Inverse Detection
            elif self.is_inverse_confirmed:
                if self.current_state not in [STATE_BLOCKED, STATE_WAIT_ROBOT, STATE_PARKING_GRAVITY, 
                                              STATE_PARKING_FINAL_MOVE, STATE_PARKING_EXIT, STATE_U_TURN_180]:
                    self.change_state(STATE_U_TURN_180)
            
            # 4. Parking Gravity Trigger
            elif self.allow_parking_logic and self.parking_counter >= 10:
                if self.current_state != STATE_PARKING_GRAVITY:
                    rospy.loginfo("Parking Spot Confirmed! Engaging Gravity.")
                    self.change_state(STATE_PARKING_GRAVITY)

            # 5. Pieton Trigger
            elif self.pieton_error > -900 and now > self.pieton_cooldown_until:
                # Exclude critical states where stopping is dangerous or confusing
                if self.current_state not in [STATE_ALIGN_PIETON, STATE_BLOCKED, STATE_U_TURN_180, STATE_WAIT_ROBOT, 
                                              STATE_INNER_ENTRY_TURN, STATE_PARKING_GRAVITY, STATE_PARKING_FINAL_MOVE, 
                                              STATE_PARKING_WAIT, STATE_PARKING_EXIT]:
                    rospy.loginfo("PIETON Detected. Stopping.")
                    
                    if self.current_state == STATE_SEARCHING_PARKING:
                        self.previous_state = STATE_SEARCHING_PARKING
                    else:
                        self.previous_state = STATE_CRUISE
                    
                    self.change_state(STATE_ALIGN_PIETON)

            # 6. Default: Cruise
            # Note: Timeout Check logic was moved to top of loop for priority
            elif self.current_state not in [STATE_U_TURN_180, STATE_ALIGN_PIETON, STATE_BLOCKED, STATE_WAIT_ROBOT,
                                            STATE_INNER_ENTRY_TURN, STATE_PARKING_GRAVITY, STATE_PARKING_FINAL_MOVE, 
                                            STATE_PARKING_WAIT, STATE_PARKING_EXIT, STATE_SEARCHING_PARKING]:
                self.current_state = STATE_CRUISE

            # ================= Execution =================
            if self.current_state == STATE_CRUISE:
                self.handle_cruise()
            elif self.current_state == STATE_SEARCHING_PARKING:
                self.handle_cruise() 
            elif self.current_state == STATE_WAIT_ROBOT:
                self.stop_robot()
            elif self.current_state == STATE_U_TURN_180:
                self.handle_timed_action(self.uturn_angular_speed, self.uturn_duration, STATE_CRUISE)
            elif self.current_state == STATE_BLOCKED:
                self.handle_timed_action(self.block_turn_speed, self.block_turn_duration, self.previous_state)
            elif self.current_state == STATE_ALIGN_PIETON:
                self.handle_pieton_align()
            elif self.current_state == STATE_INNER_ENTRY_TURN:
                # After Inner Turn, start SEARCHING PARKING
                self.handle_timed_action(self.inner_turn_speed, self.inner_turn_duration, STATE_SEARCHING_PARKING)
            elif self.current_state == STATE_PARKING_GRAVITY:
                self.handle_parking_gravity()
            elif self.current_state == STATE_PARKING_FINAL_MOVE:
                self.handle_parking_final()
            elif self.current_state == STATE_PARKING_WAIT:
                self.handle_parking_wait()
            elif self.current_state == STATE_PARKING_EXIT:
                self.handle_parking_exit()

            self.rate.sleep()

    # --- Handlers ---
    def handle_cruise(self):
        rot_cmd = -self.kp_rot * self.road_error
        rot_cmd = max(min(rot_cmd, 1.5), -1.5)
        current_linear = self.linear_speed
        if abs(self.road_error) > 35: current_linear *= 0.5
        self.publish_cmd(current_linear, rot_cmd)

    def handle_timed_action(self, angular_spd, duration, next_state):
        elapsed = (rospy.Time.now() - self.state_start_time).to_sec()
        if elapsed > duration.to_sec():
            self.change_state(next_state)
            return
        self.publish_cmd(0.0, angular_spd)

    def handle_pieton_align(self):
        elapsed = (rospy.Time.now() - self.state_start_time).to_sec()
        
        if elapsed > self.pieton_align_duration.to_sec():
            self.pieton_cooldown_until = rospy.Time.now() + self.pieton_cooldown_duration
            
            if self.mission_active:
                self.pieton_pass_count += 1
                rospy.loginfo(f"Mission Active. Pieton Pass Count: {self.pieton_pass_count}")
                
                if self.pieton_pass_count == 2:
                    rospy.loginfo("2nd Pieton Passed. Entering Inner Loop.")
                    self.pieton_pass_count = 0      
                    self.allow_parking_logic = True 
                    self.change_state(STATE_INNER_ENTRY_TURN)
                    return
            else:
                rospy.loginfo("Pieton Stop Complete (No Mission).")
            
            # Resume previous state (Cruise or Searching)
            if self.previous_state == STATE_SEARCHING_PARKING:
                self.current_state = STATE_SEARCHING_PARKING
            else:
                self.current_state = STATE_CRUISE
            return
        
        align_cmd = -self.kp_align * self.pieton_error
        if abs(self.pieton_error) > 3 and abs(align_cmd) < 0.15:
            align_cmd = 0.15 * (1 if self.pieton_error < 0 else -1)
        align_cmd = max(min(align_cmd, 0.6), -0.6)
        self.publish_cmd(0.0, align_cmd)

    def handle_parking_gravity(self):
        elapsed = (rospy.Time.now() - self.state_start_time).to_sec()
        if elapsed > self.gravity_max_duration.to_sec():
            rospy.logwarn("Parking Gravity TIMEOUT. Forcing Final Move.")
            self.change_state(STATE_PARKING_FINAL_MOVE)
            return

        if abs(self.status_flag - 7.0) < 0.1:
            rospy.loginfo("Parking Touchdown! Final Blind Move.")
            self.change_state(STATE_PARKING_FINAL_MOVE)
            return
        
        rot = 0.0
        linear = 0.088 

        # 1. Gravity Logic
        if self.parking_error > -900:
            rot = -self.kp_parking * self.parking_error
        else:
            
            linear = 0.0 
        
        # 2. Repulsion overlay
        if abs(self.road_error) > 40:
             rot += -0.005 * self.road_error

        rot = max(min(rot, 1.0), -1.0)
        self.publish_cmd(linear, rot)

    def handle_parking_final(self):
        elapsed = (rospy.Time.now() - self.state_start_time).to_sec()
        if elapsed > 7:
            self.change_state(STATE_PARKING_WAIT)
            return
        
        rot_repulsion = 0.0
        if abs(self.road_error) > 40:
            rot_repulsion = -0.005 * self.road_error 
            
        self.publish_cmd(0.05, rot_repulsion)

    def handle_parking_wait(self):
        elapsed = (rospy.Time.now() - self.state_start_time).to_sec()
        if elapsed > 2.0:
            self.change_state(STATE_PARKING_EXIT)
            return
        self.stop_robot()

    def handle_parking_exit(self):
        elapsed = (rospy.Time.now() - self.state_start_time).to_sec()
        if elapsed > self.parking_exit_duration.to_sec():
            rospy.loginfo("Parking Complete. FULL RESET.")
            self.reset_mission_vars()
            self.mission_cooldown_until = rospy.Time.now() + rospy.Duration(10.0) 
            self.change_state(STATE_CRUISE)
            return
        self.publish_cmd(0.0, self.parking_exit_speed)

    def change_state(self, new_state):
        # If entering Searching Parking state, set the deadline
        if new_state == STATE_SEARCHING_PARKING:
            rospy.loginfo("State: SEARCHING PARKING")
            self.search_end_time = rospy.Time.now() + self.parking_search_duration
        
        self.current_state = new_state
        self.state_start_time = rospy.Time.now()

    def reset_mission_vars(self):
        self.mission_active = False
        self.allow_parking_logic = False
        self.pieton_pass_count = 0
        self.flash2_counter = 0

    def publish_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear; msg.angular.z = angular
        self.pub_vel.publish(msg)

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)

if __name__ == '__main__':
    try:
        ctrl = ControleurNavig()
        ctrl.run()
    except rospy.ROSInterruptException:
        pass