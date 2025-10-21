import pygame

pygame.joystick.init() # jostick 모듈 초기화
pad = pygame.joystick.Joystick(0) # G27 데이터를 불러와 변수에 저장
pad.init() # G27 데이터 초기화

print(pad.get_name()) # >> 게임 컨트롤러의 이름, Logitech G27
print(pad.get_numaxes()) # >> 축의 개수, 5

file = open('data.txt','w')
# 조향 데이터를 로깅하기 위해 텍스트 파일을 쓰기 모드로 열기

while(True): # 반복
    print(pad.get_axis(0))
    print(pad.get_axis(1))
    print(pad.get_axis(2))
    print(pad.get_axis(3))
    

    # pygame.event.pump() # 라이브러리 내부 이벤트 발생 처리기
    # angle=pad.get_axis(0)*100 # 스티어링 휠의 축으로부터 값을 받아와 각도로 전환
    # file.write(str(angle)) # 텍스트 파일에 각도 기록
    # file.write('\n') # 텍스트 파일 줄 바꿈
    # print("조향각 : ", angle) # 조향각 출력

file.close() # 파일 닫기