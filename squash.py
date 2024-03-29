##2D Squash Game
import pygame
import time
import numpy as np
import random
import torch

class Squash():

    def __init__(self, screen_size=[400,300], paddle_width=40,
                 paddle_height=10, ball_radius=7):
        
        #Parameters
        self.screen_size = screen_size
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_radius = ball_radius
        
        #Ensure paddle and ball remain within limits of screen
        self.max_paddle_x = self.screen_size[0] - self.paddle_width
        self.max_ball_x = self.screen_size[0] - (self.ball_radius*2)
        self.max_ball_y = self.screen_size[1] - (self.ball_radius*2)
        self.paddle_y = self.screen_size[1] - self.paddle_height - 8

        self.paddle_speed = 8
        self.ball_speed = 4

        #Set up game
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Squash2D")
        self.clock = pygame.time.Clock()
        self.gameExit = False

        #States of Play
        self.BALL_IN_PADDLE = 0
        self.BALL_IN_PLAY = 1
        self.GAME_OVER = 2

        self.games = 0
        self.init_game()

    def init_game(self):
        self.score = 0
        self.reward = 0
        self.gameExit = False
        self.state = self.BALL_IN_PADDLE
        self.paddle = pygame.Rect(300, self.paddle_y, self.paddle_width,
                                  self.paddle_height)
        self.ball = pygame.Rect(300, self.paddle_y - (self.ball_radius*2),
                                self.ball_radius*2, self.ball_radius*2)
        self.ball_vel = [self.ball_speed, -self.ball_speed]

    def key_press(self, key):
        #Key press function that enables machine play
        
        if key == 'left':
            self.paddle.left -= self.paddle_speed
            if self.paddle.left < 0:
                self.paddle.left = 0

        if key == 'right':
            self.paddle.left += self.paddle_speed
            if self.paddle.left > self.max_paddle_x:
                self.paddle.left = self.max_paddle_x

        if key == 'space' and self.state == self.BALL_IN_PADDLE:
            d = random.randint(0, 1)
            if d == 0:
                self.ball_vel = [self.ball_speed, -self.ball_speed]
            else:
                self.ball_vel = [-self.ball_speed, -self.ball_speed]
            self.state = self.BALL_IN_PLAY
            
        elif key == 'return' and (self.state == self.GAME_OVER):
            self.init_game()

        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            self.gameExit = True

    def move_ball(self):
        self.ball.left += self.ball_vel[0]
        self.ball.top += self.ball_vel[1]
        
        #Bounce ball off left wall
        if self.ball.left <= 0:
            self.ball.left = 0
            self.ball_vel[0] = -self.ball_vel[0]
            
        #Bounce ball off right wall
        elif self.ball.left >= self.max_ball_x:
            self.ball.left = self.max_ball_x
            self.ball_vel[0] = -self.ball_vel[0]
        
        #Bounce ball off top wall
        if self.ball.top < 0:
            self.ball.top = 0
            self.ball_vel[1] = -self.ball_vel[1]

    def collide(self):
        #If ball collides with paddle, gain a point
        if self.ball.colliderect(self.paddle):
            self.ball.top = self.paddle_y - (self.ball_radius*2)
            self.ball_vel[1] = -self.ball_vel[1]
            self.score += 1
            self.reward += 10
        
        #If ball misses paddle, end game
        elif self.ball.top > self.paddle.top:
            self.state = self.GAME_OVER
            self.games += 1
            self.reward -= 10

    def game_step(self, action):
        #Function to take a 'game step', with action input

        #self.clock.tick()
        #self.screen.fill((0,0,0))
        
        if self.state == self.BALL_IN_PADDLE:
            self.ball.left = self.paddle.left + self.paddle.width/2 - self.ball_radius
            self.ball.top = self.paddle.top - (self.ball_radius*2)
            self.key_press('space')

        self.key_press(action)

        if self.state == self.BALL_IN_PLAY:
            self.move_ball()
            self.collide()

        elif self.state == self.GAME_OVER:
            print("GAME " + str(self.games) + " OVER.")
            self.gameExit = True

        #pygame.draw.rect(self.screen, (255,255,255), self.paddle)
        #pygame.draw.circle(self.screen, (255,255,255),
        #                   (self.ball.left + self.ball_radius, self.ball.top + self.ball_radius),
        #                   self.ball_radius)
        #pygame.display.flip()            

    def get_state(self):
        #State is a concatenated tensor of the ball coordinates, ball velocity and paddle x-coordinate
        state = torch.cat((torch.tensor([self.ball.left, self.ball.top], dtype=torch.float32),
                                       torch.tensor(self.ball_vel, dtype=torch.float32),
                                       torch.tensor([self.paddle.left], dtype=torch.float32)), dim=0)
        return state

    def set_reward(self, value):
        self.reward = value

    def get_reward(self):
        return self.reward
    
    def test_game_loop(self):
        keys = ['left', 'right', 'space', 'return']
        while self.gameExit is False:
            self.clock.tick()
            self.screen.fill((0,0,0))
            action = random.randint(0,1)
            key = keys[action]
            self.game_step(key)

            pygame.draw.rect(self.screen, (255,255,255), self.paddle)
            pygame.draw.circle(self.screen, (255,255,255),
                               (self.ball.left + self.ball_radius, self.ball.top + self.ball_radius),
                               self.ball_radius)
            pygame.display.flip()
            time.sleep(0.1)
            
    def human_game_loop(self):
        #Allow for human play to test game functionality        
        while self.gameExit is False:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()

            self.clock.tick(50)
            self.screen.fill((0,0,0))
            
            #Check for key press
            self.check_key_press()

            #Take actions depending on states
            if self.state == self.BALL_IN_PLAY:
                self.move_ball()
                self.collide()
                
            elif self.state == self.BALL_IN_PADDLE:
                self.ball.left = self.paddle.left + self.paddle.width/2 - self.ball_radius
                self.ball.top = self.paddle.top - (self.ball_radius*2)
                self.show_message("PRESS SPACE TO LAUNCH THE BALL")
            elif self.state == self.GAME_OVER:
                self.show_message("GAME OVER. PRESS ENTER TO PLAY AGAIN")

            #Draw paddle & ball
            pygame.draw.rect(self.screen, (255,255,255), self.paddle)
            pygame.draw.circle(self.screen, (255,255,255), (self.ball.left + self.ball_radius, self.ball.top + self.ball_radius), 
                               self.ball_radius)
            
            pygame.display.flip()
                        
    def check_key_press(self):
        keys = pygame.keys.get_pressed()
        
        if keys[pygame.K_LEFT]:
            key = 'left'
        
        if keys[pygame.K_RIGHT]:
            key = 'right'
        
        if keys[pygame.K_SPACE]:
            key = 'space'
        
        if keys[pygame.K_RETURN]:
            key = 'return'
            
        self.key_press(key)
        

#game = Squash()
#game.test_game_loop()
