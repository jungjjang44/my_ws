import asyncio # Python 비동기 이벤트 루프 모듈
import psycopg2 # 파이썬 DB 드라이버 (SQL 실행 등)

# PostgreSQL 서버에 SQL 연결을 맺고 세션 객체를 생성
# 현재는 로컬 PC에서 테스트용으로 사용
conn=psycopg2.connect(host="localhost", dbname="db01", user="postgres",password="root")

# Auto commit mode - listen 실행 즉시 반영됨
conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

# DB에 SQL을 날릴 커서 생성
cursor = conn.cursor()
# 명령 보내기
cursor.execute(f"LISTEN match_updates;")

# 데이터베이스 쓰기용 커넥션
wconn=psycopg2.connect(host="localhost", dbname="db01", user="postgres",password="root")
wcursor=wconn.cursor()

# 콜백 함수 (이벤트 루프가 DB 소켓에서 읽을 게 생겼을 때 호출)
def handle_notify():
    conn.poll() # 서버로부터 온 비동기 메시지를 소켓에서 끌어와서 psycopg2 내부 버퍼/큐에 반영
    for notify in conn.notifies: # 받은 알림들의 큐
        print(notify.channel, notify.payload) # [보낸 서버 프로세스 PID, 채널명, 보낸 문자열]
        wcursor.execute("INSERT INTO demo.event_log(channel, payload) VALUES (%s, %s)",(notify.channel, notify.payload))
        wconn.commit()
    conn.notifies.clear()

loop = asyncio.get_event_loop() # 
loop.add_reader(conn, handle_notify)
loop.run_forever() # DB 알림이 올 때마다 콜백이 돌고, 중단하거나 종료 로직을 구현하기 전까지는 계속 유지됨.