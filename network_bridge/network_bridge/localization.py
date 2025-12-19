#!/usr/bin/env python3
import argparse
import re
import threading
import sys
import redis
import rclpy
from rclpy.clock import Clock
from rclpy.node import Node
from custom_msgs.msg import Localization  # 아래 필드명을 가진 메시지
# float32 px,py,pz, sx,sy,sz, p_roll,p_pitch,p_yaw, s_roll,s_pitch,s_yaw

# Unity가 보낸 문자열 한 줄을 파싱하는 정규식
ROW_RE = re.compile(
    r"^(?P<ts>[^|]+)\s*\|\s*"
    r"Primary x,y,z:(?P<px>-?\d+\.?\d*),(?P<py>-?\d+\.?\d*),(?P<pz>-?\d+\.?\d*)\s*\|\s*"
    r"Secondary x,y,z:(?P<sx>-?\d+\.?\d*),(?P<sy>-?\d+\.?\d*),(?P<sz>-?\d+\.?\d*)\s*\|\s*"
    r"Primary Euler Angle:(?P<p_roll>-?\d+\.?\d*),(?P<p_pitch>-?\d+\.?\d*),(?P<p_yaw>-?\d+\.?\d*)\s*\|\s*"
    r"Secondary Euler Angle:(?P<s_roll>-?\d+\.?\d*),(?P<s_pitch>-?\d+\.?\d*),(?P<s_yaw>-?\d+\.?\d*)\s*\|\s*$"
)

def parse_payload(payload: str):
    """
    Redis payload 문자열 -> dict(float)
    매칭 실패 시 None 반환
    """
    m = ROW_RE.match(payload)
    if not m:
        return None
    g = m.groupdict()
    return {
        "px": float(g["px"]), "py": float(g["py"]), "pz": float(g["pz"]),
        "sx": float(g["sx"]), "sy": float(g["sy"]), "sz": float(g["sz"]),
        "p_roll": float(g["p_roll"]), "p_pitch": float(g["p_pitch"]), "p_yaw": float(g["p_yaw"]),
        "s_roll": float(g["s_roll"]), "s_pitch": float(g["s_pitch"]), "s_yaw": float(g["s_yaw"]),
    }

class RedisStreamToROS(Node):
    def __init__(self, host, port, password, stream, from_beginning, block_ms, count, topic):
        super().__init__("redis_stream_to_ros_localization")

        self.clock = Clock()
        # 퍼블리셔
        self.pub = self.create_publisher(Localization, topic, 10)

        # Redis 접속
        self.r = redis.Redis(host=host, port=port, password=password, decode_responses=True)

        # XREAD 설정
        self.stream = stream
        self.block_ms = block_ms
        self.count = count
        self.last_id = "0-0" if from_beginning else "$"

        # 별도 스레드에서 while 루프 형태로 소비
        self._thr = threading.Thread(target=self._reader_loop, daemon=True)
        self._thr.start()

        self.get_logger().info(
            f"Listening Redis stream='{self.stream}' from '{self.last_id}', publish to '{topic}'"
        )

    def _reader_loop(self):
        while rclpy.ok():
            try:
                resp = self.r.xread({self.stream: self.last_id}, block=self.block_ms, count=self.count)
            except Exception as e:
                self.get_logger().warn(f"Redis XREAD error: {e}")
                continue

            if not resp:
                continue

            for _, entries in resp:
                for entry_id, fields in entries:
                    # 다음 오프셋 갱신
                    self.last_id = entry_id

                    payload = fields.get("payload", "")
                    parsed = parse_payload(payload)
                    if not parsed:
                        # 파싱 실패 시 로그만 남기고 스킵
                        self.get_logger().warn(f"Parse failed: {payload}")
                        continue

                    # 메시지 채워서 publish
                    msg = Localization()
                    msg.px = (parsed["px"] + 0.25) * 14; msg.py = (parsed["py"]-0.22) * 14; msg.pz = parsed["pz"] * 14
                    msg.sx = (parsed["sx"] + 0.25) * 14; msg.sy = (parsed["sy"]-0.22) * 14; msg.sz = parsed["sz"] * 14
                    msg.p_roll = parsed["p_roll"]; msg.p_pitch = parsed["p_pitch"]; msg.p_yaw = parsed["p_yaw"]
                    msg.s_roll = parsed["s_roll"]; msg.s_pitch = parsed["s_pitch"]; msg.s_yaw = parsed["s_yaw"]
                    msg.timestamp = self.clock.now().to_msg()
                    self.pub.publish(msg)

def main():
    ap = argparse.ArgumentParser(description="Redis Streams -> ROS2 custom_msgs/Localization publisher")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6379)
    ap.add_argument("--password", default=None)
    ap.add_argument("--stream", default="unity:pose:stream")
    ap.add_argument("--from-beginning", action="store_true",
                    help="과거부터 재생(기본: 새 메시지만)")
    ap.add_argument("--block-ms", type=int, default=5000, help="XREAD 블로킹 대기(ms); 0=무한대기")
    ap.add_argument("--count", type=int, default=100, help="한 번에 읽을 최대 항목 수")
    ap.add_argument("--topic", default="/localization_info")
    args = ap.parse_args()

    rclpy.init()
    node = RedisStreamToROS(
        host=args.host, port=args.port, password=args.password,
        stream=args.stream, from_beginning=args.from_beginning,
        block_ms=args.block_ms, count=args.count, topic=args.topic
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
