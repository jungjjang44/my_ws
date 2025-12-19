import rclpy
from rclpy.node import Node
from custom_msgs.msg import Control
from std_msgs.msg import Bool, Float64
import redis


class RCControlBridge(Node):
    def __init__(self):
        super().__init__('rc_control_bridge')

        # --- Redis connection setting ---
        try:
            self.redis_client = redis.Redis(
                host='127.0.0.1',
                port=6379,
                db=0,
                decode_responses=True,
                client_name='autonomous_system'
            )
            self.get_logger().info("✅ Connected to Redis server (localhost:6379)")
        except Exception as e:
            self.get_logger().error(f"❌ Redis connection failed: {e}")
            raise e

        # --- Subscriber ---
        self.g29_sub = self.create_subscription(Control, 'control', self.g29_callback, 10)
        self.steer_sub = self.create_subscription(Float64, '/control/steer', self.steer_callback, 10)
        self.pedal_sub = self.create_subscription(Float64, '/control/pedal', self.pedal_callback, 10)
        self.mode_subscription2 = self.create_subscription(Bool, '/logitech/contorlmode', self.mode_callback, 10)

        # 전송 카운터
        self.count = 0

        # --- control parameter ---
        self.steer = 0.0
        self.pedal = 0.0
        self.is_auto = False

        self.get_logger().info("🚗 RCControlBridge node started. Waiting for /control messages...")

    def steer_callback(self, msg:Float64):
        self.steer = round(msg.data, 3)

    def pedal_callback(self, msg:Float64):
        self.pedal = round(msg.data, 3)
    
    def mode_callback(self, msg:Bool):
        self.is_auto = msg.data

    def g29_callback(self, msg:Control):
        self.count += 1
        self.count = self.count % 255

        steering = round(msg.lat_axis, 3)
        ap_bp = round(msg.long_axis, 3)
        gear = int(msg.gear)

        payload_steer, payload_ap_bp = 0.0, 0.0

        if self.is_auto:
            payload_steer = -self.steer
            payload_ap_bp = self.pedal
        else:
            payload_steer = steering
            payload_ap_bp = ap_bp

        # Redis는 문자열(str)만 가능하므로 CSV 포맷으로 변환
        payload = f"{self.count}, {payload_steer}, {payload_ap_bp}, {gear}"             

        try:
            self.redis_client.publish("control_data", payload)
            self.get_logger().info(f"📤 Sent to Redis: {payload}")
        except Exception as e:
            self.get_logger().error(f"Redis publish failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RCControlBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
