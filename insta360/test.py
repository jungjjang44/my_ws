import cv2,time,signal,sys

DEV = "/dev/video0"
W,H,FPS=2880,1440,30

cap=None
running=True

def cleanup():
    global cap
    if cap is not None:
        try:
            cap.release()
        except:
            pass
    try:
        cv2.destroyAllWindows()
    except:
        pass
    time.sleep(0.2)

def handle_sigint(sig,frame):
    global running
    runnning=False

signal.signal(signal.SIGINT,handle_sigint)

try:
    cap=cv2.VideoCapture(DEV,cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Camera can't open")
    
    # 기본 설정
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    time.sleep(0.5)
    for _ in range(10): cap.read()

    while running:
        ok,frame=cap.read()
        if not ok:
            time.sleep(0.02)
            continue

        cv2.imshow("Insta360 X5",frame)
        key=cv2.waitKey(1)&0xFF
        if key==27:
            break
finally:
    cleanup()
