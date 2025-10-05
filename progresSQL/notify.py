import time
import psycopg2

# PostgreSQL 서버에 SQL 연결을 맺고 세션 객체를 생성
# 현재는 로컬 PC에서 테스트용으로 사용
conn=psycopg2.connect(host="localhost", dbname="db01", user="postgres",password="root")

# DB에 SQL을 날릴 커서 생성
cursor = conn.cursor()
conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

while True:
    val=time.time()
    cursor.execute(f"NOTIFY match_updates, '{val}';")
    time.sleep(5)


'''
0. 현재 listen.py와 notify.py는 같은 DB에 연결되어 있음.
1. 현재 시간에 해당하는 메시지를 지정한 채널로 이벤트를 날림 (엄밀히 바로 DB에 저장하는 것이 X)
2. listen.py에서 받은 메시지를 DB에 저장하는 중
3. 추후 메시지 형태를 jpeg 확장자 형태로 변경해서 다시 보내야 함
4. 현재 로컬 PC에서만 동작하는데, 추후 원격에서 돌리도록 변경해야 함
'''

'''
실행 방법
Terminal 1: 
cd my_ws/progresSQL
python3 listen.py

Terminal 2:
cd my_ws/progresSQL
python3 notify.py

Terminal 3:
krri@krri:~$ sudo -i -u postgres

postgres@krri:~$ psql -h localhost -U postgres -d db01
비번: root

db01=# SELECT id, channel, payload, received_at
FROM demo.event_log
ORDER BY id DESC
LIMIT 10;

'''