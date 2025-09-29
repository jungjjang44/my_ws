import psycopg2

sql="""
CREATE SCHEMA IF NOT EXISTS demo;
CREATE TABLE IF NOT EXISTS demo.event_log (
    id BIGSERIAL PRIMARY KEY,
    channel TEXT NOT NULL,
    payload TEXT NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_event_log_time ON demo.event_log(received_at);
"""

conn=psycopg2.connect(host="localhost", dbname="db01", user="postgres",password="root")
cur=conn.cursor()
cur.execute(sql)
conn.commit()
cur.close()
conn.close()