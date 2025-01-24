import random
from enum import Enum
import pygame
import sys
from os import path

class RobotAction(Enum):
    LEFT=0
    DOWN=1
    RIGHT=2
    UP=3

class GridTile(Enum):
    _FLOOR=0
    ROBOT=1
    TARGET=2

    def __str__(self):
        return self.name[:1]

class Robot:

    def __init__(self, grid_rows=63, grid_cols=63, fps=1):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.reset()

        self.fps = fps
        self.last_action=''
        self._init_pygame()

    def _init_pygame(self):
        pygame.init()
        pygame.display.init()

        self.clock = pygame.time.Clock()

        self.action_font = pygame.font.SysFont("Calibre",10)
        self.action_info_height = self.action_font.get_height()

        self.cell_height = 12
        self.cell_width = 12
        self.cell_size = (self.cell_width, self.cell_height)        

        self.window_size = (self.cell_width * self.grid_cols, self.cell_height * self.grid_rows + self.action_info_height)

        self.window_surface = pygame.display.set_mode(self.window_size) 

        file_name = path.join(path.dirname(__file__), "sprites/bot.png")
        img = pygame.image.load(file_name)
        self.robot_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/floor.png")
        img = pygame.image.load(file_name)
        self.floor_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/target.png")
        img = pygame.image.load(file_name)
        self.goal_img = pygame.transform.scale(img, self.cell_size) 


    def reset(self, seed=None):
        self.robot_pos = [0,0]

        random.seed(seed)
        self.target_pos = [
            random.randint(1, self.grid_rows-1),
            random.randint(1, self.grid_cols-1)
        ]

    def perform_action(self, robot_action:RobotAction) -> bool:
        self.last_action = robot_action

        if robot_action == RobotAction.LEFT:
            if self.robot_pos[1]>0:
                self.robot_pos[1]-=1
        elif robot_action == RobotAction.RIGHT:
            if self.robot_pos[1]<self.grid_cols-1:
                self.robot_pos[1]+=1
        elif robot_action == RobotAction.UP:
            if self.robot_pos[0]>0:
                self.robot_pos[0]-=1
        elif robot_action == RobotAction.DOWN:
            if self.robot_pos[0]<self.grid_rows-1:
                self.robot_pos[0]+=1

        return self.robot_pos == self.target_pos

    def render(self):
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):

                if([r,c] == self.robot_pos):
                    print(GridTile.ROBOT, end=' ')
                elif([r,c] == self.target_pos):
                    print(GridTile.TARGET, end=' ')
                else:
                    print(GridTile._FLOOR, end=' ')

            print()
        print()

        self._process_events()

        self.window_surface.fill((255,255,255))

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                
                pos = (c * self.cell_width, r * self.cell_height)
                self.window_surface.blit(self.floor_img, pos)

                if([r,c] == self.target_pos):
                    self.window_surface.blit(self.goal_img, pos)

                if([r,c] == self.robot_pos):
                    self.window_surface.blit(self.robot_img, pos)
                
        text_img = self.action_font.render(f'Action: {self.last_action}', True, (0,0,0), (255,255,255))
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text_img, text_pos)       

        pygame.display.update()
                
        self.clock.tick(self.fps)  

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if(event.type == pygame.KEYDOWN):
                if(event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                


# For unit testing
if __name__=="__main__":
    Robot = Robot()
    Robot.render()

    while(True):
        rand_action = random.choice(list(RobotAction))
        print(rand_action)

        Robot.perform_action(rand_action)
        Robot.render()