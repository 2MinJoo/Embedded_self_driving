import RPi.GPIO as GPIO
import time
import pygame

pygame.init()

screen = pygame.display.set_mode((500, 300))

pygame.display.set_caption("Key Event")
clock = pygame.time.Clock()
run = True

acc = 0
dir = 0


left_pwm = 6
left_dir = 13
right_pwm = 19
right_dir = 26

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(left_pwm, GPIO.OUT)
GPIO.setup(left_dir, GPIO.OUT)
GPIO.setup(right_pwm, GPIO.OUT)
GPIO.setup(right_dir, GPIO.OUT)

GPIO.output(left_dir, False)
GPIO.output(right_dir, True)

left_p = GPIO.PWM(left_pwm, 100)
left_p.start(0)
right_p = GPIO.PWM(right_pwm, 100)
right_p.start(0)

try:
    while run:
        # pygame.event.get() : 키를 눌렀을때 이벤트
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # esc 누르면 종
                    run = False

        # pygame.key.get_pressed() - 전체 키배열중 현재 눌려져있는 키를 bool형식의 튜플로 반환
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            dir = dir - 1
        elif keys[pygame.K_RIGHT]:
            dir = dir + 1
        elif keys[pygame.K_UP]:
            acc = acc + 1
        elif keys[pygame.K_DOWN]:
            acc = acc - 1

        if acc < 0 :
            acc = 0
        if acc > 10 :
            acc = 10

        if dir < -5 :
            dir = -5
        if dir > 5 :
            dir = 5

        left_pwm_v = (acc * 10) + (dir * -5)
        if left_pwm_v < 0:
            left_pwm_v = 0
        if left_pwm_v > 100:
            left_pwm_v = 100
        if left_pwm_v > 0 & left_pwm_v < 50 :
            left_pwm_v = 50

        right_pwm_v = (acc * 10) + (dir * 5)
        if right_pwm_v < 0:
            right_pwm_v = 0
        if right_pwm_v > 100:
            right_pwm_v = 100
        if right_pwm_v > 0 & right_pwm_v < 50 :
            right_pwm_v = 50

        left_p.ChangeDutyCycle( left_pwm_v)
        right_p.ChangeDutyCycle(right_pwm_v)



        time.sleep(0.03)

        screen.fill(pygame.color.Color(255, 255, 255))
        pygame.display.flip()
    clock.tick(60)
    pygame.quit()
except KeyboardInterrupt:
    left_p.ChangeDutyCycle(0)
