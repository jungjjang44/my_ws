import redis, time, json, os
from datetime import datetime

# 지금은 로컬 PC를 서버로 사용함. 추후에 서버 IP+포트번호+DB 번호를 입력해주면 됨. (DB번호는 초기값 0)
REDIS_URL=os.getenv("REDIS_URL","redis://localhost:6379/0")

# 사용할 스트림(키) 설정, 기본값은 items
# 스트림: 시간 순서대로 쌓이는 메시지 큐
STREAM=os.getenv("STREAM","items")

# Redis 클라이언트 생성 (decode_responses=바이트가 아니라 문자열로 주고 받는다.)
r=redis.Redis.from_url(REDIS_URL, decode_responses=True)

i=0
while True:
    # payload: 보낼 메시지 본문 구성 (단순 증가값+데모 문자열+UTC 시각)
    payload={"id":i,"content":f"Hello #{i}","ts":datetime.utcnow().isoformat()}
    
    # 스트림에 한 항목 추가 XADD 형태(필드-값 쌍으로 넣어야 함) + JSON 문자열로 저장
    msg_id=r.xadd(STREAM, {"payload":json.dumps(payload)})
    print(f"[XADD] stream={STREAM} id={msg_id} payload={payload}")
    i+=1
    time.sleep(2)