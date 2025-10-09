import cv2, time, signal

IDX = 1            # 열리는 인덱스
W, H, FPS = 1920, 1080, 30   # 16:9로 Flat 뷰 유도

running = True
def handle(sig,frm):
    global running; running=False
signal.signal(signal.SIGINT, handle)

cap = cv2.VideoCapture(IDX, cv2.CAP_DSHOW)   # ★ MSMF 대신 DSHOW
if not cap.isOpened():
    raise RuntimeError("open fail")

# 먼저 최소 설정만 적용 (MJPG/1080p/30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS,          FPS)

time.sleep(0.5)
for _ in range(10): cap.read()

print("actual:", cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                cap.get(cv2.CAP_PROP_FPS))

while running:
    ok, frame = cap.read()
    if not ok: 
        time.sleep(0.01); continue
    cv2.imshow("X5 Flat try (DSHOW, 1080p)", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release(); cv2.destroyAllWindows()
