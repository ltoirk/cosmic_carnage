from constants import *
import numpy as np
import pygame
import random
from helper_utils import *


class GameObject:
    def __init__(self, x, y, boundary_x, boundary_y, object_type):
        self.object_type = object_type
        self.x = x
        self.y = y
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y
        self.destroyed = False
        
    def step(self):
        pass
    
    def destroy(self):
        self.destroyed = True
        self.x = OUTSIDE_WORLD_EDGE
        self.y = OUTSIDE_WORLD_EDGE
        


class Ship(GameObject):
    def __init__(self, x, y, type, team, boundary_x=1500, boundary_y=1500, ship_id=0):
        super().__init__(x, y, boundary_x, boundary_y, 'ship')
        self.type = type
        self.points = 0
        self.boosts = {'attack': 0, 'defense': 0}
        self.team = team
        self.boost_duration = {'attack': 0, 'defense': 0}
        self.last_shot = 0
        self.last_supporter_heal = 0
        self.hit_box = SHIP_HITBOXES[self.type]
        self.ship_id = ship_id
        self.rotation = np.random.uniform(0, 360)
        self.v_x = 0
        self.v_y = 0
        

        if self.type in SHIP_TYPE_STATS:
            for key, value in SHIP_TYPE_STATS[self.type].items():
                setattr(self, key, value)

        self.max_health = self.health
        self.attack_speed = BULLET_SPEED

    def move(self, thrust, rotation):  # [-1, 1]
        self.rotation -= rotation * rotation_factor
        thrust *= self.speed
        self.v_x = -thrust * np.sin(self.rotation * np.pi / 180)
        self.v_y = -thrust * np.cos(self.rotation * np.pi / 180)

        self.x += self.v_x * MOVEMENT_FACTOR
        self.y += self.v_y * MOVEMENT_FACTOR

        self.x = max(0, min(self.x, self.boundary_x - SHIP_SCALE - 1))
        self.y = max(0, min(self.y, self.boundary_y - SHIP_SCALE - 1))

    def take_damage(self, damage):
        net_defense = min(self.defense * (1 + self.boosts['defense']), damage * 0.8)  # max 80% damage reduction
        self.health -= (damage - net_defense)
        if self.health <= 0:
            self.destroy()

    def heal(self, heal_amount):
        self.health += heal_amount
        self.health = min(self.health, self.max_health)

    def boost(self, boost, duration, type):
        self.boosts[type] = boost
        self.boost_duration[type] += duration

    def shoot(self):
        if self.last_shot > 0:
            return None
        self.last_shot = self.attack_rate
        bullet_v_x = -self.attack_speed * np.sin(self.rotation * np.pi / 180)
        bullet_v_y = -self.attack_speed * np.cos(self.rotation * np.pi / 180)
        net_attack = self.attack * (1 + self.boosts['attack'])
        return Bullet(self.x, self.y, bullet_v_x, bullet_v_y, net_attack, self.team, self.boundary_x, self.boundary_y, self.ship_id)

    def step(self, thrust, rotation, shoot):  # [-1, 1], [-1, 1], [0, 1]
        # assert -1 <= thrust  and thrust <= 1
        # assert -1 <= rotation and rotation <= 1
        # assert -1 <= shoot and shoot <= 1
        ### clip the values to be within the range
        thrust = np.clip(thrust, -1, 1)
        rotation = np.clip(rotation, -1, 1)
        shoot = np.clip(shoot, 0, 1)

        shoot = np.round(shoot) >= 0

        self.last_shot = max(0, self.last_shot - 1)
        self.last_supporter_heal = max(0, self.last_supporter_heal - 1)
        
        for boost_type in self.boosts:
            self.boost_duration[boost_type] = max(0, self.boost_duration[boost_type] - 1)
            if self.boost_duration[boost_type] <= 0:
                self.boosts[boost_type] = 0

        self.move(thrust, rotation)
        if shoot:
            return self.shoot()

        return None
    
    def get_obs(self):
        return [self.x, self.y, self.rotation, self.health, int(self.boosts['attack'] > 0), int(self.boosts['defense'] > 0)]


class Boost(GameObject):
    def __init__(self, boundary_x, boundary_y):
        super().__init__(np.random.uniform(0, boundary_x*0.7), np.random.uniform(0, boundary_y*0.7), boundary_x, boundary_y, 'boost')
        self.type = random.choice(['attack', 'defense', 'health', 'coins'])
        self.duration = BOOST_DURATION
        self.hit_box = SHIP_HITBOXES[ATTACKER_TYPE]
        
    def pickup(self, ship):
        if self.type == 'attack':
            ship.boost(ATTACK_BOOST, self.duration, 'attack')
        elif self.type == 'defense':
            ship.boost(DEFENSE_BOOST, self.duration, 'defense')
        elif self.type == 'health':
            ship.heal(HEALTH_BOOST)
            
        self.destroy()


class Coins(GameObject):
    def __init__(self, boundary_x, boundary_y):
        super().__init__(np.random.uniform(0, boundary_x*0.7), np.random.uniform(0, boundary_y*0.7), boundary_x, boundary_y, 'coins')
        self.hit_box = SHIP_HITBOXES[ATTACKER_TYPE]
        self.type = 'coins'
    def pickup(self, ship):
        ship.points += AGENT_REWARD['coin']
        self.destroy()


class Bullet(GameObject):
    def __init__(self, x, y, v_x, v_y, attack, team, boundary_x=1500, boundary_y=1500, ship_id=0):
        super().__init__(x, y, boundary_x, boundary_y, 'bullet')
        self.team = team
        self.v_x = v_x
        self.v_y = v_y
        self.attack = attack
        self.hit_box = BULLET_HITBOX
        self.ship_id = ship_id
        
    def step(self):
        self.x += self.v_x * MOVEMENT_FACTOR
        self.y += self.v_y * MOVEMENT_FACTOR

        if self.x < 0 or self.x > self.boundary_x or self.y < 0 or self.y > self.boundary_y:
            self.destroy()
            


class Asteroid(GameObject):
    def __init__(self, boundary_x, boundary_y):
        super().__init__(np.random.uniform(0, boundary_x), np.random.uniform(0, boundary_y), boundary_x, boundary_y, 'asteroid')
        self.type = random.choice([1, 2, 3, 4, 5])
        
        if np.random.uniform() > 0.5:
            self.x = random.choice([0, boundary_x])  # spawn at leftmost or rightmost edge
            self.y = np.random.uniform(0, boundary_y)
        else:
            self.x = np.random.uniform(0, boundary_x)
            self.y = random.choice([0, boundary_y])  # spawn at topmost or bottommost edge
            
        self.v_x = np.random.uniform(-1, 1)
        self.v_y = np.random.uniform(-1, 1)
        
        self.duration = 0
        
        self.size = np.random.uniform(20, 100)  # 20 to 100% dimension
        self.hit_box = ASTEROID_HITBOXES[self.type]
        
    def step(self):
        self.duration += 1
        if (self.duration > ASTEROID_LIFETIME) or (self.x < 0 or self.x > self.boundary_x or self.y < 0 or self.y > self.boundary_y):
            self.destroy()
            
        self.x += self.v_x * MOVEMENT_FACTOR
        self.y += self.v_y * MOVEMENT_FACTOR


def check_collision(obj1, obj2):
    distance = np.linalg.norm(np.array([obj1.x, obj1.y]) - np.array([obj2.x, obj2.y]))
    if distance <= (obj1.hit_box + obj2.hit_box)/1.2:
        # print("Collision detected between", obj1.object_type, obj2.object_type)
        return True
    return False

def render_ship(screen, ship, color_code_list):
    x, y = ship.x, ship.y
    rotation = ship.rotation
    ## triangle is speeder, square is attacker, pentagon is defender, circle is support
    
    surface = None
    if ship.type == ATTACKER_TYPE:
        ship_img = SHIP_PRELOADED_AGENT.get(ATTACKER_TYPE).get(str(color_code_list[ship.team]))
    elif ship.type == SPEEDSTER_TYPE:
        ship_img = SHIP_PRELOADED_AGENT.get(SPEEDSTER_TYPE).get(str(color_code_list[ship.team]))
    elif ship.type == DEFENDER_TYPE:
        ship_img = SHIP_PRELOADED_AGENT.get(DEFENDER_TYPE).get(str(color_code_list[ship.team]))
    elif ship.type == SUPPORT_TYPE:
        ship_img = SHIP_PRELOADED_AGENT.get(SUPPORT_TYPE).get(str(color_code_list[ship.team]))
        ### draw a circle for support heal range with shaded area alpha 10
        pygame.draw.circle(screen, pygame.Color(color_code_list[ship.team]), (int(x), int(y)), HEAL_RANGE, 2)
        surface = pygame.Surface((HEAL_RANGE*2, HEAL_RANGE*2), pygame.SRCALPHA)
        pygame.draw.circle(surface, pygame.Color(list(color_code_list[ship.team]) + [50]), (HEAL_RANGE, HEAL_RANGE), HEAL_RANGE)
        
    # print("Team", ship.team, rotation)
    # fill(ship_img, pygame.Color(color_code_list[ship.team]))
    # pygame.draw.rect(ship_img,  pygame.Color(color_code_list[ship.team]), ship_img.get_rect(), 2)
        
    # ship_img = pygame.transform.scale(ship_img, (SHIP_SCALE, SHIP_SCALE))
    ship_img = pygame.transform.rotate(ship_img, rotation)
    screen.blit(ship_img, (x-ship_img.get_width()//2, y-ship_img.get_height()//2))
    if surface is not None:
        screen.blit(surface, (x-HEAL_RANGE, y-HEAL_RANGE))
    
def render_bullet(screen, bullet, color_code_list):
    x, y = bullet.x, bullet.y
    bullet_img = BULLET_PRELOADED_AGENT[str(color_code_list[bullet.team])]

    # fill(bullet_img, pygame.Color(color_code_list[bullet.team]))
    # pygame.draw.rect(bullet_img, pygame.Color(color_code_list[bullet.team]), bullet_img.get_rect(), 2)
   
    screen.blit(bullet_img, (x, y))
    
def render_asteroid(screen, asteroid):
    x, y = asteroid.x, asteroid.y
    asteroid_img = ASTEROID_PRELOADED[asteroid.type]
    # pygame.draw.rect(asteroid_img, pygame.Color('blue'), asteroid_img.get_rect(), 2)
    screen.blit(asteroid_img, (x, y))

def render_boost(screen, boost):
    x, y = boost.x, boost.y
    boost_img = PICKUP_PRELOADED[boost.type]
    screen.blit(boost_img, (x, y))
        

