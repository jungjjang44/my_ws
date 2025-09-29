import redis, os, json

REDIS_URL=os.getenv("REDIS_URL","redis://localhost:6379/0")
STREAM=os.getenv("STREAM","items")

# 그룹: 
GROUP=os.getenv("GROUP","g1")

# 컨슈머: 
CONSUMER=os.getenv("CONSUMER","c1")

r=redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Consumer Group 생성
try:
    r.xgroup_create(STREAM,GROUP,id="$",mkstream=True)
    print(f"[GROUP] created '{GROUP}' on '{STREAM}'")
except redis.exceptions.ResponseError as e:
    if "BUSYGROUP" not in str(e):
        raise

print(f"[READ] stream={STREAM} group={GROUP} consumer={CONSUMER}")
while True:
    # 새로운 메시지를 최대 5초 대기하며 10개씩 읽기
    # 그룹, 컨슈머는 스트림을 처리하는 협업 단위라고 생각하면 편할 듯
    resp=r.xreadgroup(GROUP,CONSUMER,{STREAM:">"}, count=10, block=5000)
    if not resp:
        continue
    # 새로운 메시지 들어온 것이 있으면 메시지 ID와 필드-값 딕셔너리 가져오기
    for stream,events in resp:
        for msg_id, fields in events:

            # JSON 역직렬화
            data=json.loads(fields["payload"])
            print(f"[CONSUME] id={msg_id} data={data}")
            
            # 처리 완료 ACK (이 메시지는 성공 처리함을 그룹에 알림)
            # pending 목록에서 제거함
            r.xack(STREAM,GROUP,msg_id)

'''
Terminal 1: cd my_ws/redis/writer_stream.py
Terminal 2: cd my_ws/redis/consumer_group.py
python 코드 실행 (2개 터미널 모두)

Terminal 3: 
>> redis_cli
>> XRANGE items - + COUNT 20 (20개 로그 보겠다??)

'''