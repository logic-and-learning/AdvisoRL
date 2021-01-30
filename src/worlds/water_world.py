if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../')

from worlds.game_objects import Actions
import random, math, os, pickle
import numpy as np

"""
Auxiliary class with the configuration parameters that the Game class needs
"""
class WaterWorldParams:
    def __init__(self, state_file = None, max_x = 1000, max_y = 700, b_num_colors = 6, 
                       b_radius = 20, b_velocity = 30, b_num_per_color = 10,
                       use_velocities = True, ball_disappear = True):
        self.max_x = max_x
        self.max_y = max_y
        self.b_num_colors = b_num_colors
        self.b_radius = b_radius
        self.b_velocity = b_velocity
        self.a_vel_delta = b_velocity
        self.a_vel_max = 3*b_velocity
        self.b_num_per_color = b_num_per_color
        self.state_file = state_file
        self.use_velocities = use_velocities
        self.ball_disappear = ball_disappear
        
class WaterWorld:

    def __init__(self, params):
        self.params = params
        self.use_velocities = params.use_velocities
        self._load_map()
        if params.state_file is not None:
            self.load_state(params.state_file)
        self.env_game_over = False
        # Setting up event detectors
        self.current_collisions_old = set()
        self._update_events()

    def _get_current_collision(self):
        ret = set()
        for b in self.balls:
            if self.agent.is_colliding(b):
                ret.add(b)
        return ret        

    def _update_events(self):
        self.true_props = ""
        current_collisions = self._get_current_collision()
        for b in current_collisions - self.current_collisions_old:
            self.true_props += b.color
        self.current_collisions_old = current_collisions        

    def execute_action(self, a, elapsedTime=0.1):
        action = Actions(a)
        # computing events
        self._update_events()

        # if balls disappear, then relocate balls that the agent is colliding before the action
        if self.params.ball_disappear:
            for b in self.balls:
                if self.agent.is_colliding(b):
                    pos, vel = self._get_pos_vel_new_ball()
                    b.update(pos, vel)

        # updating the agents velocity
        self.agent.execute_action(action)
        balls_all = [self.agent] + self.balls
        max_x, max_y = self.params.max_x, self.params.max_y

        # updating position
        for b in balls_all:
            b.update_position(elapsedTime)
        
        # handling collisions
        for i in range(len(balls_all)):
            b = balls_all[i]
            # walls
            if b.pos[0] - b.radius < 0 or b.pos[0] + b.radius > max_x:
                # Place ball against edge
                if b.pos[0] - b.radius < 0: b.pos[0] = b.radius          
                else: b.pos[0] = max_x - b.radius
                # Reverse direction
                b.vel = b.vel * np.array([-1.0,1.0])
            if b.pos[1] - b.radius < 0 or b.pos[1] + b.radius > max_y:
                # Place ball against edge
                if b.pos[1] - b.radius < 0: b.pos[1] = b.radius
                else: b.pos[1] = max_y - b.radius
                # Reverse directio
                b.vel = b.vel * np.array([1.0,-1.0])
        

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.agent.get_actions()

    def get_state(self):
        return None # we are only using "simple reward machines" for the craft domain

    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        return self.true_props

    # The following methods return different feature representations of the map ------------
    def get_features(self):
        #_,features = self._get_features_Vis()
        _,features = self._get_features_HER()
        return features


    def _get_features_Vis(self):
        vel_max   = float(self.params.a_vel_max)
        range_max = (self.params.max_x**2+self.params.max_y**2)**0.5
        max_x     = self.params.max_x
        max_y     = self.params.max_y
        radius    = self.params.b_radius    
        agent     = self.agent
        a_x, a_y  = agent.pos[0], agent.pos[1]

        # The state space is even larger and continuous: 
        # The agent has 30 eye sensors pointing in all 
        # directions and in each direction is observes 
        # 5 variables: the range, the type of sensed object (green, red), 
        # and the velocity of the sensed object. 
        # The agent's proprioception includes two additional sensors for 
        # its own speed in both x and y directions. 
        # This is a total of 152-dimensional state space.
        # map from object classes to numbers
        num_eyes = 16 # in practice, each eye goes to both sides
        num_classes = self.params.b_num_colors + 1 # I'm including the walls here

        # adding walls
        contact_points = {}
        for i in range(num_eyes):
            # features per eye: range, type, v_x, v_y
            angle_pos = i * 180 / num_eyes
            angle_neg = angle_pos + 180
            
            # walls collisions
            col_pos = []
            col_neg = []
            if angle_pos == 0:
                col_pos.append(np.array([max_x, a_y]))
                col_neg.append(np.array([0, a_y]))
            elif angle_pos == 90:
                col_pos.append(np.array([a_x, max_y]))
                col_neg.append(np.array([a_x, 0]))
            else:
                m = math.tan(math.radians(angle_pos))
                c = a_y - m * a_x
                w_n = np.array([(max_y - c)/m, max_y])
                w_e = np.array([max_x, m*max_x + c])
                w_w = np.array([0.0, c])
                w_s = np.array([-c/m, 0.0])
                if angle_pos < 90:
                    col_pos.extend([w_n, w_e])
                    col_neg.extend([w_s, w_w])
                else:
                    col_pos.extend([w_n, w_w])
                    col_neg.extend([w_s, w_e])
            
            # adding the points
            for p in col_pos: add_contact_point(contact_points, angle_pos, (dist(agent.pos,p),p,'W'))
            for p in col_neg: add_contact_point(contact_points, angle_neg, (dist(agent.pos,p),p,'W'))


        # Adding balls
        for b in self.balls:
            if agent.is_colliding(b):
                continue
            # computing the eyes that collide with this ball
            dd = dist(agent.pos, b.pos)
            theta = math.degrees(math.asin(b.radius/dd))
            dx, dy = b.pos[0] - a_x, b.pos[1] - a_y
            alpha = normalize_angle(math.degrees(math.atan2(dy, dx)))
            alpha_plus  = alpha + theta
            alpha_minus = alpha - theta
            if alpha_minus < 0:
                alpha_minus += 360
                alpha_plus += 360
            i =  math.ceil((num_eyes * alpha_minus)/180)
            angle = i * 180 / num_eyes
            while angle <= alpha_plus:
                angle_real = normalize_angle(angle)
                # checking that the ball is in the rigth range
                if dd-b.radius < contact_points[angle_real][0]:
                    p, q, r = b.pos[0], b.pos[1], b.radius                    
                    if angle_real in [90, 270]:
                        dis = r**2 - (a_x-p)**2
                        if dis < 0: # the line misses the ball
                            print("It missed the ball?")
                        else: # the line intersects the circle (in one or two points)
                            for case in [-1,1]:
                                x_p = a_x
                                y_p = q+case*dis**0.5
                                c_p = np.array([x_p,y_p])
                                add_contact_point(contact_points, angle_real, (dist(agent.pos,c_p),c_p,b))
                    else:
                        m = math.tan(math.radians(angle_real))
                        c = a_y - m * a_x
                        A = m**2+1
                        B = 2*(m*c-m*q-p)
                        C = q**2-r**2+p**2-2*c*q+c**2
                        dis = B**2-4*A*C
                        if dis < 0: # the line misses the ball
                            print("It missed the ball?", alpha, theta, alpha_minus, angle, alpha_plus)
                        else: # the line intersects the circle (in one or two points)
                            for case in [-1,1]:
                                x_p = (-B+case*dis**0.5)/(2*A)
                                y_p = m*x_p+c
                                c_p = np.array([x_p,y_p])
                                add_contact_point(contact_points, angle_real, (dist(agent.pos,c_p),c_p,b))
                i += 1
                angle = i * 180 / num_eyes
                        
        # range, type, v_x, v_y
        n_features_per_eye = 3+num_classes
        n_features = n_features_per_eye*2*num_eyes+2
        features = np.zeros(n_features,dtype=np.float)
        colliding_points = []
        for i in range(2*num_eyes):
            # features per eye: range, type, v_x, v_y
            dd, p, obj = contact_points[i * 180 / num_eyes]
            colliding_points.append(p)
            features[i*n_features_per_eye:(i+1)*n_features_per_eye] = get_eye_features(dd, obj, num_classes, range_max, vel_max)
        # adding the agents velocity
        features[n_features-2:n_features] = agent.vel / vel_max

        return colliding_points, features

    def _get_features_HER(self):
        # Absolute position and velocity of the anget + relative positions and velocities of the other balls
        # with respect to the agent
        if self.use_velocities:
            agent, balls = self.agent, self.balls
            n_features = 4 + len(balls) * 4
            features = np.zeros(n_features,dtype=np.float)

            pos_max = np.array([float(self.params.max_x), float(self.params.max_y)])
            vel_max = float(self.params.b_velocity + self.params.a_vel_max)

            features[0:2] = agent.pos/pos_max
            features[2:4] = agent.vel/float(self.params.a_vel_max)
            for i in range(len(balls)):
                # If the balls are colliding, I'll not include them 
                # (because there us nothing that the agent can do about it)
                b = balls[i]
                if not self.params.ball_disappear or not agent.is_colliding(b):
                    init = 4*(i+1)
                    features[init:init+2]   = (b.pos - agent.pos)/pos_max
                    features[init+2:init+4] = (b.vel - agent.vel)/vel_max
        else:
            agent, balls = self.agent, self.balls
            n_features = 4 + len(balls) * 2
            features = np.zeros(n_features,dtype=np.float)

            pos_max = np.array([float(self.params.max_x), float(self.params.max_y)])
            vel_max = float(self.params.b_velocity + self.params.a_vel_max)

            features[0:2] = agent.pos/pos_max
            features[2:4] = agent.vel/float(self.params.a_vel_max)
            for i in range(len(balls)):
                # If the balls are colliding, I'll not include them 
                # (because there us nothing that the agent can do about it)
                b = balls[i]
                if not self.params.ball_disappear or not agent.is_colliding(b):
                    init = 2*i + 4
                    features[init:init+2]   = (b.pos - agent.pos)/pos_max

        return [], features
        #return [b.pos for b in balls if not agent.is_colliding(b)], features
    

    def _is_collising(self, pos):
        for b in self.balls + [self.agent]:
            if np.linalg.norm(b.pos - np.array(pos), ord=2) < 2*self.params.b_radius:
                return True
        return False

    def _get_pos_vel_new_ball(self):
        max_x    = self.params.max_x
        max_y     = self.params.max_y
        radius    = self.params.b_radius
        b_vel     = self.params.b_velocity
        angle = random.random()*2*math.pi
        if self.use_velocities:
            vel = b_vel*math.sin(angle),b_vel*math.cos(angle)
        else:
            vel = 0.0, 0.0
        while True:
            pos = 2*radius + random.random()*(max_x - 2*radius), 2*radius + random.random()*(max_y - 2*radius)
            if not self._is_collising(pos) and np.linalg.norm(self.agent.pos - np.array(pos), ord=2) > 4*radius:
                break
        return pos, vel    

    # The following methods create the map ----------------------------------------------

    def _load_map(self):
        # contains all the actions that the agent can perform
        actions = [Actions.up.value, Actions.left.value, Actions.right.value, Actions.down.value, Actions.none.value]
        max_x    = self.params.max_x
        max_y     = self.params.max_y
        radius    = self.params.b_radius
        b_vel     = self.params.b_velocity
        vel_delta = self.params.a_vel_delta
        vel_max   = self.params.a_vel_max
        # Adding the agent
        pos_a = [2*radius + random.random()*(max_x - 2*radius), 2*radius + random.random()*(max_y - 2*radius)]
        self.agent = BallAgent("A", radius, pos_a, [0.0,0.0], actions, vel_delta, vel_max)
        # Adding the balls
        self.balls = []
        colors = "abcdefghijklmnopqrstuvwxyz"
        for c in range(self.params.b_num_colors):
            for _ in range(self.params.b_num_per_color):
                color = colors[c]
                pos, vel = self._get_pos_vel_new_ball()
                ball = Ball(color, radius, pos, vel)
                self.balls.append(ball)

    def save_state(self, filename):
        # Saves the agent and balls positions and velocities
        with open(filename, 'wb') as output:
            pickle.dump(self.agent, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.balls, output, pickle.HIGHEST_PROTOCOL)

    def load_state(self, filename):
        # Load the agent and balls positions and velocities
        with open(filename, 'rb') as input:
            self.agent = pickle.load(input)
            self.balls = pickle.load(input)
        if not self.use_velocities:
            # Removing balls velocities
            for b in self.balls:
                b.vel = np.array([0.0,0.0], dtype=np.float)

def normalize_angle(alpha):
    while not(0 <= alpha < 360): 
        if alpha < 0:
            alpha += 360
        if alpha >= 360:
            alpha -= 360
    return alpha

def add_contact_point(contact_points, angle, new_point):
    if angle not in contact_points:
        contact_points[angle] = new_point
    elif new_point[0] < contact_points[angle][0]:
        contact_points[angle] = new_point

def get_eye_features(dd, obj, num_classes, range_max, vel_max):
    # range, type, v_x, v_y
    n_features = 1+2+num_classes
    ret = np.zeros(n_features,dtype=np.float)
    ret[0] = dd / range_max
    ret[1:3] = [0.0,0.0] if obj == "W" else obj.vel / vel_max
    type_id = -1 if obj == "W" else ord(obj.color) - ord('a') + 3
    ret[type_id] = 1
    return ret

def dist(p1,p2):
    ret = np.linalg.norm(p1-p2, ord=2)
    if type(ret) != np.float64:
        print("Error, the distance is not a float")
        print("p1", p1)
        print("p2", p2)
        print("ret", ret)        
    return ret
    
class Ball:
    def __init__(self, color, radius, pos, vel): #row and column
        self.color = color
        self.radius = radius
        self.update(pos, vel)

    def __str__(self):
        return "\t".join([self.color, str(self.pos[0]), str(self.pos[1]), str(self.vel[0]), str(self.vel[1])])

    def update_position(self, elapsedTime):
        self.pos = self.pos + elapsedTime * self.vel

    def update(self, pos, vel):
        self.pos = np.array(pos, dtype=np.float)
        self.vel = np.array(vel, dtype=np.float)

    def is_colliding(self, ball):
        d = np.linalg.norm(self.pos - ball.pos, ord=2)
        return d <= self.radius + ball.radius

class BallAgent(Ball):
    def __init__(self,color, radius, pos, vel, actions, vel_delta, vel_max):
        super().__init__(color, radius, pos, vel)
        self.reward  = 0
        self.actions = actions
        self.vel_delta = float(vel_delta)
        self.vel_max = float(vel_max)

    def execute_action(self, action):
        # updating velocity
        delta = np.array([0,0])
        if action == Actions.up:    delta = np.array([0.0,+1.0])
        if action == Actions.down:  delta = np.array([0.0,-1.0])
        if action == Actions.left:  delta = np.array([-1.0,0.0])
        if action == Actions.right: delta = np.array([+1.0,0.0])
        self.vel += self.vel_delta * delta
        
        # checking limits
        self.vel = np.clip(self.vel, -self.vel_max, self.vel_max)

    def get_actions(self):
        return self.actions


def get_position(b, max_y):
    return int(round(b.pos[0])), int(max_y) - int(round(b.pos[1]))

def draw_ball(b, colors, thickness, gameDisplay, pygame, max_y):
    pygame.draw.circle(gameDisplay, colors[b.color], get_position(b, max_y), b.radius, thickness)

def draw_point(a_pos, pos, gameDisplay, pygame, max_y):
    a_pos_real = int(round(a_pos[0])), int(max_y) - int(round(a_pos[1]))
    pos_real = int(round(pos[0])), int(max_y) - int(round(pos[1]))
    pygame.draw.line(gameDisplay, (0,0,0), a_pos_real, pos_real)
    pygame.draw.circle(gameDisplay, (255,0,0), pos_real, 4)


def get_colors():
    colors = {}
    colors["A"] = (0,0,0)
    colors["a"] = (255,0,0)
    colors["b"] = (0,255,0)
    colors["c"] = (0,0,255)
    colors["d"] = (255,255,0) # yellow
    colors["e"] = (0,255,255) # cyan
    colors["f"] = (255,0,255) # magenta
    colors["g"] = (192,192,192)
    colors["h"] = (128,128,128)
    colors["i"] = (128,0,0)
    colors["j"] = (128,128,0)
    colors["k"] = (0,128,0)
    colors["l"] = (128,0,128)
    colors["m"] = (0,128,128)
    colors["n"] = (0,0,128)
    return colors

def play():
    import pygame, time
    from reward_machines.reward_machine import RewardMachine

    from tester.tester import Tester
    from tester.tester_params import TestingParameters    
    from qrm.learning_params import LearningParameters

    # hack: moving one directory up (to keep relative references to ./src)
    import os
    os.chdir("../")

    tester = Tester(LearningParameters(), TestingParameters(), "../experiments/water/tests/water_7.txt")
    if tester is None:
        task = "../experiments/water/reward_machines/t1.txt"
        state_file = "../experiments/water/maps/world_0.pkl"
        max_x = 400
        max_y = 400
        b_num_per_color = 2
        b_radius = 15
        use_velocities = True
        ball_disappear = False

        params = WaterWorldParams(state_file, b_radius=b_radius, max_x=max_x, max_y=max_y, 
                                  b_num_per_color=b_num_per_color, use_velocities = use_velocities, 
                                  ball_disappear=ball_disappear)
    else:
        task   = tester.get_task_rms()[-2]
        params = tester.get_task_params(task).game_params

    max_x, max_y = params.max_x, params.max_y

    game = WaterWorld(params)    
    rm = RewardMachine(task) 
    s1 = game.get_state()
    u1 = rm.get_initial_state()

    print("actions", game.get_actions())

    pygame.init()
    
    black = (0,0,0)
    white = (255,255,255)
    colors = get_colors()
    
    gameDisplay = pygame.display.set_mode((max_x, max_y))
    pygame.display.set_caption('Water world :)')
    clock = pygame.time.Clock()
    crashed = False

    t_previous = time.time()
    actions = set()
    while not crashed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
            if event.type == pygame.KEYUP:
                if Actions.left in actions and event.key == pygame.K_LEFT:
                    actions.remove(Actions.left)
                if Actions.right in actions and event.key == pygame.K_RIGHT:
                    actions.remove(Actions.right)
                if Actions.up in actions and event.key == pygame.K_UP:
                    actions.remove(Actions.up)
                if Actions.down in actions and event.key == pygame.K_DOWN:
                    actions.remove(Actions.down)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    actions.add(Actions.left)
                if event.key == pygame.K_RIGHT:
                    actions.add(Actions.right)
                if event.key == pygame.K_UP:
                    actions.add(Actions.up)
                if event.key == pygame.K_DOWN:
                    actions.add(Actions.down)
            

        t_current = time.time()
        t_delta = (t_current - t_previous)

        # Getting the action
        if len(actions) == 0: a = Actions.none
        else: a = random.choice(list(actions))

        # Executing the action
        game.execute_action(a.value, t_delta)

        s2 = game.get_state()
        events = game.get_true_propositions()
        u2 = rm.get_next_state(u1, events)
        reward = rm.get_reward(u1,u2,s1,a,s2)

        # printing image
        gameDisplay.fill(white)
        for b in game.balls:
            draw_ball(b, colors, 0, gameDisplay, pygame, max_y)
        draw_ball(game.agent, colors, 3, gameDisplay, pygame, max_y)
        pygame.display.update()
        clock.tick(20)

        # print info related to the task
        if reward > 0: print("REWARD!! ----------------!------------!")
        if rm.is_terminal_state(u2): 
            print("Machine state:", u2, "(terminal)")
        else:
            print("Machine state:", u2)

        t_previous = t_current
        s1, u1 = s2, u2

    pygame.quit()

def save_random_world(num_worlds, folder_out="../../experiments/water/maps/"):
    max_x = 400
    max_y = 400
    b_num_per_color = 2
    b_radius = 15
    use_velocities = True
    params = WaterWorldParams(None, b_radius=b_radius, max_x=max_x, max_y=max_y, b_num_per_color=b_num_per_color, use_velocities = use_velocities)
    
    for i in range(num_worlds):
        random.seed(i)
        game = WaterWorld(params)
        game.save_state("%sworld_%d.pkl"%(folder_out,i))
        

# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    play()
    #save_random_world(11)