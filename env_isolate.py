import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from game_objects import *
from constants import *
from helper_utils import *
import gc

np.random.seed(SEED)
random.seed(SEED)

class MultiAgentSpaceShooterEnv(gym.Env):
    def __init__(self, num_agents=3, fleet_size=6, max_fps=5, asteroid_count=15, boost_count=5, coin_count=10, render_mode='human', max_steps=1000, obs_config={'enemy_ships': 3, 'asteroids': 1, 'boosts': 1, 'coins': 1, 'bullets': 3}, img_obs=False):
        super(MultiAgentSpaceShooterEnv, self).__init__()
        self.seed(SEED)
        self.num_agents = num_agents
        self.fleet_size = fleet_size
        self.render_mode = render_mode
        self.asteroid_count = asteroid_count
        self.boost_count = boost_count
        self.coin_count = coin_count
        self.max_fps = max_fps
        self.img_obs = img_obs
        
        self.max_steps = max_steps
        self.obs_config = obs_config

        self.screen_width = WORLD_SIZE
        self.screen_height = self.screen_width
        
        if not self.img_obs:
            self.color_code_list = generate_rgb_values(num_agents)
        else:
            self.ALLY_COLOR = (0, 255, 0)
            self.ENEMY_COLOR = (255, 0, 0)
            self.color_code_list = [self.ALLY_COLOR] + [self.ENEMY_COLOR]*(num_agents-1)
        
        for color in self.color_code_list:
            for ship in SHIP_PRELOADED_AGENT:
                ship_img = SHIP_PRELOADED[ship].copy()
                #fill(ship_img, pygame.Color(color))
                SHIP_PRELOADED_AGENT[ship][str(color)] = ship_img
                
            bullet_img = BULLET_PRELOADED.copy()
            #fill(bullet_img, pygame.Color(color))
            BULLET_PRELOADED_AGENT[str(color)] = bullet_img
                
        
        if self.render_mode == 'human' or self.img_obs:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)
            
        print(SHIP_PRELOADED_AGENT)
        self.reset()
        low = np.array([-1, -1, -1]*self.num_agents*self.fleet_size, dtype=np.float32)
        high = np.array([1, 1, 1]*self.num_agents*self.fleet_size, dtype=np.float32)

        self.action_space = gym.spaces.box.Box(low=low, high=high, shape=(self.num_agents*self.fleet_size*3,), dtype=np.float32)
        
        if not self.img_obs:
            self.low, self.high = [], []

            ### ally ship
            self.low = [0, 0, 0, 0, 0, 0]*self.fleet_size
            self.high = [WORLD_SIZE, WORLD_SIZE, 1, 100, 1, 1]*self.fleet_size
            ally_ship = len(self.low)
            ### enemy ship
            considered_ships = min(self.fleet_size*self.obs_config['enemy_ships'], self.fleet_size*(self.num_agents-1))
            self.low.extend([0, 0, 0]*considered_ships)
            self.high.extend([WORLD_SIZE, WORLD_SIZE, 100]*considered_ships)


            considered_others = self.fleet_size*(self.obs_config['asteroids']+self.obs_config['boosts']+self.obs_config['coins']+self.obs_config['bullets'])
            self.low.extend([0, 0]*considered_others)
            self.high.extend([WORLD_SIZE, WORLD_SIZE]*considered_others)

            self.low, self.high = np.asarray(self.low*self.num_agents), np.asarray(self.high*self.num_agents)

            print("Observation space per agent ally ship:",ally_ship, "enemy ship:", considered_ships, "others:", considered_others, "Total:", self.low.shape[0], sep="\n")
            self.observation_space = gym.spaces.box.Box(low=self.low, high=self.high, dtype=np.float32)
            
        else:
            self.observation_space =  gym.spaces.box.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3), dtype=np.uint8)
            
        print("Observation space shape",self.get_obs().shape, type(self.observation_space))
        print("Action space",self.action_space, type(self.action_space))

  
    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
    
    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        gc.collect()
        base_ship_types = [ATTACKER_TYPE, SPEEDSTER_TYPE, DEFENDER_TYPE, SUPPORT_TYPE][:self.fleet_size]
        self.agents = [base_ship_types+random.sample(base_ship_types, self.fleet_size-len(base_ship_types)) for _ in range(self.num_agents)] ## min 4 different types of ships will be present in the fleet, rest are sampled. in future it will be customized.
        self.agent_score = [0]*self.num_agents
        ## spawn agent's fleets in equidistant locations dynamically based on varying number of agents in extreme opposite. each fleet's ship should spawn in a circle at that fleet's location
        screen_half_diagonal = self.screen_width*0.75/2 ##80% of the screen width//2
    
        fleet_locations = [[np.cos(2*np.pi*i/self.num_agents)*screen_half_diagonal, np.sin(2*np.pi*i/self.num_agents)*screen_half_diagonal] for i in range(self.num_agents)]
        ### add noise to the fleet locations by 5% of the screen width
        fleet_locations = [[x + np.random.uniform(-0.05*self.screen_width, 0.05*self.screen_width), y + np.random.uniform(-0.05*self.screen_height, 0.05*self.screen_height)] for x, y in fleet_locations]
        
        ### displace the fleet locations to the center of the screen
        fleet_locations = [[self.screen_width//2 + x, self.screen_height//2 + y] for x, y in fleet_locations]
        
        ### clip the fleet locations to the screen
        fleet_locations = [[np.clip(x, 0, self.screen_width), np.clip(y, 0, self.screen_height)] for x, y in fleet_locations]
        
        
        self.agent = []
        for i, fleet in enumerate(self.agents):
            ### each ship spawn in a circle at the fleet's location
            self.agent.append([Ship(fleet_locations[i][0] + 100*np.cos(2*np.pi*j/self.fleet_size), fleet_locations[i][1] + 100*np.sin(2*np.pi*j/self.fleet_size), fleet[j], i, boundary_x=self.screen_height, boundary_y=self.screen_width, ship_id=j) for j in range(self.fleet_size)])
            
        self.bullets = []
        self.asteroids = [Asteroid(self.screen_width, self.screen_height) for _ in range(self.asteroid_count)]
        self.boosts = [Boost(self.screen_width, self.screen_height) for _ in range(self.boost_count)]
        self.coins = [Coins(self.screen_width, self.screen_height) for _ in range(self.coin_count)]
        
  
        ### Monitoring stats
        self.dmg_matrix = np.zeros((self.num_agents*self.fleet_size, self.num_agents*self.fleet_size))
        self.score_matrix = np.zeros((self.num_agents*self.fleet_size))
        self.hit_score_matrix = np.zeros((self.num_agents*self.fleet_size))
        self.asteroid_dmage_matrix = np.zeros((self.num_agents*self.fleet_size))
        self.supp_health_matrix = np.zeros((self.num_agents*self.fleet_size, self.num_agents*self.fleet_size))
        self.boosts_matrix = np.zeros((self.num_agents*self.fleet_size))
        self.alive_agents = [self.fleet_size]*self.num_agents
        self.health_agents = []
        for fleet in self.agent:
            self.health_agents +=[ship.health for ship in fleet]
            
        self.current_step = 0
        self.last_step = self.current_step - 1
        if not self.img_obs:
            self.calc_dist_matrix()
        return self.get_obs(), {}
        

    def calc_dist_matrix(self):
        ## cache the distance matrix
        if self.last_step == self.current_step:
            return
        
        
        ships = []
        for i in self.agent:
            ships.extend(i)
            
        self.last_step = self.current_step
        self.dist_matrix = self.calc_object_dist_matrix(ships)
        self.asteroid_dist_matrix = self.calc_object_dist_matrix(self.asteroids)
        self.boost_dist_matrix = self.calc_object_dist_matrix(self.boosts)
        self.coin_dist_matrix = self.calc_object_dist_matrix(self.coins)
        self.bullets_dist_matrix = self.calc_object_dist_matrix(self.bullets)
        

    def calc_object_dist_matrix(self, obj_list):
        dist_matrix = np.zeros((self.num_agents*self.fleet_size, len(obj_list)))
        for i, fleet in enumerate(self.agent):
            for j, ship in enumerate(fleet):
                for k, obj in enumerate(obj_list):
                    dist_matrix[i*self.fleet_size+j][k] = np.abs(ship.x-obj.x) + np.abs(ship.y-obj.y)
        return dist_matrix

    def get_obs(self):
        if self.img_obs:
            if self.current_step == 0:
                self.render()
            if self.current_step != self.last_step:
                color_code_list = self.color_code_list.copy()
                self.state = []
                for i, fleet in enumerate(self.agent):
                    self.color_code_list[i] = self.ALLY_COLOR
                    self.color_code_list[:i] = [self.ENEMY_COLOR]*(i)
                    self.color_code_list[i+1:] = [self.ENEMY_COLOR]*(self.num_agents-i-1)
                    self.render()
                    state = np.transpose(pygame.surfarray.array3d(pygame.display.get_surface()), (1, 0, 2))
                    # import cv2
                    # cv2.imwrite(f"{i}.png", state)
                    self.state.append(state)
                    
                self.color_code_list = color_code_list
                ############### TODO ################
                ##### Do color inversion so that ships coloured blue are the current agent's ships
                self.last_step = self.current_step
                
            return np.asarray(self.state)

        obs = []
        self.calc_dist_matrix()
        
        for fleet in self.agent:
            dist_matrix = self.dist_matrix.copy()
            ship = fleet[0]
            dist_matrix[ship.team*self.fleet_size:(ship.team+1)*self.fleet_size, ship.team*self.fleet_size:(ship.team+1)*self.fleet_size] = np.inf
            
            # print("Dist Matrix",dist_matrix, sep="\n")
        
            fleet_obs = []
            for ship in fleet:
                fleet_obs.extend(ship.get_obs()) #6*4 = 24
                
            ### for each ship choose closest 1 asteroid, 3 unique enemy ship, 1 boost, 1 coin => 4*(3+1+1+1) = 48 coordinates + 4*3 boost yes/no = 60
            
            enemy_ship_location_ids = []
            max_iter = 100
            while len(enemy_ship_location_ids) < min(self.obs_config['enemy_ships']*self.fleet_size, sum(self.alive_agents) - self.alive_agents[ship.team]): ### we are trying to make it unique for each ship
                for ship in fleet:
                    
                    enemy_ids = sorted(range(self.num_agents*self.fleet_size), key=lambda x: dist_matrix[ship.team*self.fleet_size+ship.ship_id][x])
                    enemy_ship_location_ids.extend(enemy_ids[:self.obs_config['enemy_ships']])
                    enemy_ship_location_ids = list(set(enemy_ship_location_ids))
                    for enemy_id in enemy_ids[:self.obs_config['enemy_ships']]:
                        dist_matrix[:,enemy_id] = np.inf
                max_iter -= 1
                
                if max_iter == 0:
                    ## duplicate enemy ships
                    while len(enemy_ship_location_ids) < min(self.obs_config['enemy_ships']*self.fleet_size, sum(self.alive_agents) - self.alive_agents[ship.team]):
                        needed = min(self.obs_config['enemy_ships']*self.fleet_size, sum(self.alive_agents) - self.alive_agents[ship.team])
                        enemy_ship_location_ids.extend(enemy_ship_location_ids[:needed-len(enemy_ship_location_ids)])
                    print("!!!!WARNING: MAX ITERATION REACHED!!!!")
                    break
                    
    
                        
            for enemy_id in enemy_ship_location_ids[:min(self.obs_config['enemy_ships']*self.fleet_size, self.fleet_size*(self.num_agents-1))]:
                ship = self.agent[enemy_id//self.fleet_size][enemy_id%self.fleet_size]
                fleet_obs.extend([ship.x, ship.y, ship.health])
                
            coin_coords = np.ones((self.fleet_size*self.obs_config['coins'], 2))*OUTSIDE_WORLD_EDGE
            boost_coords = np.ones((self.fleet_size*self.obs_config['boosts'], 2))*OUTSIDE_WORLD_EDGE
            asteroid_coords = np.ones((self.fleet_size*self.obs_config['asteroids'], 2))*OUTSIDE_WORLD_EDGE
            bullet_coords = np.ones((self.fleet_size*self.obs_config['bullets'], 2))*OUTSIDE_WORLD_EDGE
            
            bullet_count = 0
            for i, ship in enumerate(fleet): ### we are not trying to make it unique for each ship
                asteroid_dist = np.argsort(self.asteroid_dist_matrix[ship.team*self.fleet_size+ship.ship_id])
                boost_dist = np.argsort(self.boost_dist_matrix[ship.team*self.fleet_size+ship.ship_id])
                coin_dist = np.argsort(self.coin_dist_matrix[ship.team*self.fleet_size+ship.ship_id])
                bullet_dist = np.argsort(self.bullets_dist_matrix[ship.team*self.fleet_size+ship.ship_id])
                
                ast_coord = [[self.asteroids[asteroid_dist[j]].x, self.asteroids[asteroid_dist[j]].y] for j in range(min(self.obs_config['asteroids'], len(asteroid_dist)))]
                ast_coord+= [[OUTSIDE_WORLD_EDGE, OUTSIDE_WORLD_EDGE]]*(self.obs_config['asteroids']-len(ast_coord))
                asteroid_coords[i*self.obs_config['asteroids']:(i+1)*self.obs_config['asteroids']] = ast_coord
                
                boost_coord = [[self.boosts[boost_dist[j]].x, self.boosts[boost_dist[j]].y] for j in range(min(self.obs_config['boosts'], len(boost_dist)))]
                boost_coord+= [[OUTSIDE_WORLD_EDGE, OUTSIDE_WORLD_EDGE]]*(self.obs_config['boosts']-len(boost_coord))
                boost_coords[i*self.obs_config['boosts']:(i+1)*self.obs_config['boosts']] = boost_coord
                
                coin_coord = [[self.coins[coin_dist[j]].x, self.coins[coin_dist[j]].y] for j in range(min(self.obs_config['coins'], len(coin_dist)))]
                coin_coord+= [[OUTSIDE_WORLD_EDGE, OUTSIDE_WORLD_EDGE]]*(self.obs_config['coins']-len(coin_coord))
                coin_coords[i*self.obs_config['coins']:(i+1)*self.obs_config['coins']] = coin_coord
                
                close_bullet = [[self.bullets[bullet_dist[j]].x, self.bullets[bullet_dist[j]].y] for j in range(min(3, len(bullet_dist)))]
                close_bullet+= [[OUTSIDE_WORLD_EDGE, OUTSIDE_WORLD_EDGE]]*(3-len(close_bullet))
                bullet_coords[bullet_count:bullet_count+len(close_bullet)] = close_bullet
                bullet_count += len(close_bullet)
            
            fleet_obs.extend(asteroid_coords.flatten()) ## 4*2 = 8
            fleet_obs.extend(boost_coords.flatten()) ## 4*2 = 8
            fleet_obs.extend(coin_coords.flatten()) ## 4*2 = 8
            fleet_obs.extend(bullet_coords.flatten()) ## 4*3*2 = 24
            
            obs.extend(fleet_obs) ## 60+48 = 108
           
           
         
        return np.asarray(obs) ## num_agents*108
            


    def step(self, actions): ## thrust, rotation, shoot
        self.current_step += 1
        ### actions -> agent1, agent2, agent3, ...
        actions = np.array(actions)
                        
        for obj in self.bullets + self.asteroids:
            obj.step()

        current_reward = np.asarray([0]*self.num_agents*self.fleet_size)
        done = False
        
        for i, fleet in enumerate(self.agent):
            for j, ship in enumerate(fleet):
                assert i == ship.team
                assert j == ship.ship_id
                
                if ship.destroyed: ## already destroyed
                    continue
        
                temp_boost = False
                ### heal support ships
                if (ship.type != SUPPORT_TYPE) and (ship.last_supporter_heal == 0):
                    for ship_ in  self.agent[i]:
                        if ship_.type != SUPPORT_TYPE: ## not a support ship
                            continue
                        if (ship_.team != ship.team) or (ship_.ship_id != ship.ship_id): ## not the same ship
                            continue
                        distance = np.linalg.norm(np.array([ship.x, ship.y]) - np.array([ship_.x, ship_.y]))
                        if distance < HEAL_RANGE:
                            temp_boost = True
                            ship.heal(ship_.heal_strength)
                            current_reward[ship_.team*self.fleet_size+ship_.ship_id] += AGENT_REWARD['heal']
                            ship.last_supporter_heal = ship_.heal_every
                            self.supp_health_matrix[ship_.team*self.fleet_size+ship_.ship_id][ship.team*self.fleet_size+ship.ship_id] += ship_.heal_strength
                            break
                                
                ## movement and shooting
                if temp_boost:
                    ship.boosts['attack'] = max(ATTACK_BOOST/5, ship.boosts['attack']) ## heal ship that are not boosted already
                    ship.boosts['defense'] = max(DEFENSE_BOOST/5, ship.boosts['defense'])
                    
                thrust, rotation, shoot = actions[ship.team*self.fleet_size*3+ship.ship_id*3:ship.team*self.fleet_size*3+ship.ship_id*3+3]
                bullet = ship.step(thrust, rotation, shoot)
                if bullet is not None:
                    self.bullets.append(bullet)
                    
                
                ### damage matrix
                for asteroid in self.asteroids:
                    if check_collision(asteroid, ship):
                        ship.take_damage(ASTEROID_DMAGE)
                        self.asteroid_dmage_matrix[ship.team*self.fleet_size+ship.ship_id] += ASTEROID_DMAGE
                        asteroid.destroy()
                        self.asteroids.remove(asteroid)
                        ### del asteroid
                        del asteroid
                        current_reward[ship.team*self.fleet_size+ship.ship_id] += AGENT_REWARD['being_hit']
                        
                ## check for boosts
                for boost in self.boosts:
                    if check_collision(boost, ship):
                        boost.pickup(ship)
                        self.boosts_matrix[ship.team*self.fleet_size+ship.ship_id] += 1
                        boost.destroy()
                        self.boosts.remove(boost)
                        del boost
                        current_reward[ship.team*self.fleet_size+ship.ship_id] += AGENT_REWARD['boost']
                        
                ## check for coins
                for coin in self.coins:
                    if check_collision(coin, ship):
                        coin.pickup(ship)
                        self.agent_score[i] += AGENT_REWARD['coin']
                        self.score_matrix[ship.team*self.fleet_size+ship.ship_id] += 10
                        coin.destroy()
                        self.coins.remove(coin)
                        del coin
                        current_reward[ship.team*self.fleet_size+ship.ship_id] += AGENT_REWARD['coin']
                
                ## check for bullet collision  
                for bullet in self.bullets:
                    for asteroid in self.asteroids:
                        if check_collision(bullet, asteroid):
                            asteroid.destroy()
                            self.asteroids.remove(asteroid)
                            self.bullets.remove(bullet)
                            current_reward[bullet.team*self.fleet_size+bullet.ship_id] += AGENT_REWARD['asteroid_hit']
                            bullet.destroy()
                            del asteroid
                            break
                    
                    if bullet.destroyed:
                        del bullet
                        continue
                    if bullet.team == ship.team:
                        continue
                    
                    if check_collision(bullet, ship):
                        ship.take_damage(bullet.attack)
                        self.dmg_matrix[bullet.team*self.fleet_size+bullet.ship_id][ship.team*self.fleet_size+ship.ship_id] += bullet.attack
                        self.agent_score[bullet.team] += AGENT_REWARD['hit']
                        self.hit_score_matrix[bullet.team*self.fleet_size+bullet.ship_id] += AGENT_REWARD['hit']
                        current_reward[bullet.team*self.fleet_size+bullet.ship_id] += AGENT_REWARD['hit']
                        bullet.destroy()
                        self.bullets.remove(bullet)
                        current_reward[ship.team*self.fleet_size+ship.ship_id] += AGENT_REWARD['being_hit']
                        
                        if ship.destroyed:
                            current_reward[ship.team*self.fleet_size+ship.ship_id] += AGENT_REWARD['being_killed'] - AGENT_REWARD['ally_died']
                            current_reward[ship.team*self.fleet_size:(ship.team+1)*self.fleet_size] += AGENT_REWARD['ally_died']
                            current_reward[bullet.team*self.fleet_size+bullet.ship_id] += AGENT_REWARD['kill']
                            self.agent_score[bullet.team] += AGENT_REWARD['kill']
                            
                        del bullet
                        continue
                           
                        
                ## check for ship collision, we don't remove the ship, just mark it as destroyed
                if ship.destroyed:
                    self.alive_agents[ship.team] -= 1
                    current_reward[ship.team*self.fleet_size+ship.ship_id] += AGENT_REWARD['being_killed']
                    
                if temp_boost:
                    if ship.attack_boost != ATTACK_BOOST: ## already boosted
                        ship.attack_boost = 0
                    if ship.defense_boost != DEFENSE_BOOST: ## already boosted
                        ship.defense_boost = 0
                    
                self.health_agents[ship.team*self.fleet_size+ship.ship_id] = ship.health
                            

        ### spawn new objects
        for obj_list, obj_class, count in [
            (self.asteroids, Asteroid, self.asteroid_count),
            (self.boosts, Boost, self.boost_count),
            (self.coins, Coins, self.coin_count)
        ]:
            if len(obj_list) < 0.5 * count:
                for _ in range(count - len(obj_list)):
                    obj_list.append(obj_class(self.screen_width, self.screen_height))
                
        ### remove destroyed bullets
        for obj in self.bullets: 
            if obj.destroyed:
                self.bullets.remove(obj)
                del obj
        

        if self.render_mode == 'human':
            self.render()
            
        alive = 0
        alive_teams = -1
        for team_id, i in enumerate(self.alive_agents):
            assert i >= 0
            if i > 0:
                alive += 1
                alive_teams = team_id
        
        ### if 1 or 0 agents are alive, end the episode
        done = True
        if alive>=2:
            done = False
        elif alive == 1:
            if self.num_agents > 1: ### atleast 2 agents are present in the game
                current_reward[alive_teams*self.fleet_size:(alive_teams+1)*self.fleet_size] += AGENT_REWARD['win'] ## game ends when only 1 team is alive
            else:
                done = False
        else:
            done = True
                
        trunc = True if self.current_step >= self.max_steps else False

        if self.num_agents == 1: #### single agent combined reward
            current_reward = sum(current_reward)
            return self.get_obs(), current_reward, done, trunc, {}
        epsilon = 1e-8
        return self.get_obs(), sum(current_reward),done, trunc, {}
        

    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))
        for bullet in self.bullets:
            render_bullet(self.screen, bullet, self.color_code_list)
            
        for asteroid in self.asteroids:
            render_asteroid(self.screen, asteroid)
            
        for boost in self.boosts:
            render_boost(self.screen, boost)
        
        for coin in self.coins:
            render_boost(self.screen, coin)
            
        for i, fleet in enumerate(self.agent):
            for j, ship in enumerate(fleet):
                if ship.destroyed:
                    continue
                render_ship(self.screen, ship, self.color_code_list)
            
        if self.render_mode == 'human':
            pygame.display.flip()
        # pygame.display.flip()
        # self.clock.tick(self.max_fps)  # Limit to max FPS

    def close(self):
        pygame.quit()
        

if __name__ == "__main__":
    import time
    start = time.time()
    num_agents = 3
    max_fps = 120
    mode = 'human'
    img_obs = False
    env = MultiAgentSpaceShooterEnv(num_agents=num_agents, fleet_size=1, max_fps=max_fps, asteroid_count=0, boost_count=0, coin_count=0, render_mode=mode, max_steps=1000, img_obs=img_obs)
    done = False
    trunc = False
    step = 0

    reward = 0
    reward_total = 0
    while not done and not trunc:
        # action = env.action_space.sample()
        step+=1
        ### clear the output screen
        # print("\033[H\033[J")        
        obs = env.get_obs()
        print("Step: ",step, obs.shape)
        # print("Observation",obs.shape[0]/(num_agents))
        reward_total += reward
        print("Reward",reward, "Total Reward",reward_total)
        # print("Agent Score",env.agent_score)
        # print("Alive Agents",env.alive_agents)
        # print("Health Agents",env.health_agents)
        # print("DMG Matrix",env.dmg_matrix)
        # print("Score Matrix",env.score_matrix)
        # print("Heal Matrix",env.supp_health_matrix)

        ### add pygame input
        done = False
        rotate = 0
        thrust = 0
        shoot = 0
        
        
        action_i = [thrust, rotate, shoot]
        
        ##### USER INPUT
        if mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                keys = pygame.key.get_pressed()
                if keys[pygame.K_ESCAPE]:
                    done = True
                if keys[pygame.K_a]:
                    rotate = -1
                if keys[pygame.K_d]:
                    rotate = 1
                if keys[pygame.K_w]:
                    thrust = 1
                if keys[pygame.K_s]:
                    thrust = -1
                if keys[pygame.K_SPACE]:
                    shoot = 1
            
            if time.time() - start > 1/(2*max_fps):
                action_i = [thrust, rotate, shoot]
                start = time.time()
        
        action = env.action_space.sample()
        action[:3] = action_i
        # print(env.bullets)
        # obs, reward, done, trunc, _ = env.step(action)
        start = time.time()
        if num_agents>1:
            obs, reward, done, trunc, _ = env.step(action)
        else:
            obs, reward, done, trunc, _ = env.step(action)
            # time.sleep(0.1)
        print("Step Time",time.time()-start)
        if done:
            print("Episode truncated at step", step)
            time.sleep(2)
            env.reset()
            trunc = False
            done = False
        
        
    env.close()