from omegaconf import OmegaConf
config = OmegaConf.load('config.yaml')

SEED = config.SEED

ATTACKER_TYPE = config.ATTACKER_TYPE
SPEEDSTER_TYPE = config.SPEEDSTER_TYPE
DEFENDER_TYPE = config.DEFENDER_TYPE
SUPPORT_TYPE = config.SUPPORT_TYPE
WORLD_SIZE = config.WORLD_SIZE
OUTSIDE_WORLD_EDGE = config.OUTSIDE_WORLD_EDGE

ASTEROID_LIFETIME = config.ASTEROID_LIFETIME
BOOST_DURATION = config.BOOST_DURATION
ASTEROID_DMAGE = config.ASTEROID_DAMAGE
HEAL_RANGE = config.HEAL_RANGE

ASTEROID_SCALE = config.ASTEROID_SCALE
SHIP_SCALE = config.SHIP_SCALE
BULLET_SCALE = config.BULLET_SCALE

AGENT_REWARD = config.AGENT_REWARD

MOVEMENT_FACTOR = config.MOVEMENT_FACTOR
rotation_factor = config.rotation_factor
BULLET_SPEED = config.BULLET_SPEED
SHIP_TYPE_STATS = OmegaConf.to_container(config.SHIP_TYPE_STATS)

ATTACK_BOOST = config.ATTACK_BOOST
DEFENSE_BOOST = config.DEFENSE_BOOST
HEALTH_BOOST = config.HEALTH_BOOST


ATTACKER_PATH = '/home/linux/Downloads/Cosmic-Carnage-master/assets/ship/attacker.png'
SPEEDSTER_PATH = '/home/linux/Downloads/Cosmic-Carnage-master/assets/ship/speedster.png'
DEFENDER_PATH = '/home/linux/Downloads/Cosmic-Carnage-master/assets/ship/defender.png'
SUPPORT_PATH = '/home/linux/Downloads/Cosmic-Carnage-master/assets/ship/support.png'
BULLET_PATH = '/home/linux/Downloads/Cosmic-Carnage-master/assets/bullet.png'
ASTEROID_PATH = '/home/linux/Downloads/Cosmic-Carnage-master/assets/Asteroid/'
PICKUP_PATH = '/home/linux/Downloads/Cosmic-Carnage-master/assets/pickup/'

import pygame
## get hitbox of the image

SHIP_HITBOXES = {}
ASTEROID_HITBOXES = {}
PICKUP_HITBOXES = {}
ship_paths = {
	ATTACKER_TYPE: ATTACKER_PATH,
	SPEEDSTER_TYPE: SPEEDSTER_PATH,
	DEFENDER_TYPE: DEFENDER_PATH,
	SUPPORT_TYPE: SUPPORT_PATH
}
SHIP_PRELOADED = {}
SHIP_PRELOADED_AGENT = {}
ASTEROID_PRELOADED = {}
PICKUP_PRELOADED = {}

for ship_type, path in ship_paths.items():
	image = pygame.image.load(path)
	image = pygame.transform.scale(image, (SHIP_SCALE, SHIP_SCALE))
	hitbox = max(image.get_width(), image.get_height())
	SHIP_HITBOXES[ship_type] = hitbox
	SHIP_PRELOADED[ship_type] = image
	SHIP_PRELOADED_AGENT[ship_type] = {}
	
 
for asteroid_type in range(1, 6):
	image = pygame.image.load(f'{ASTEROID_PATH}{asteroid_type}.png')
	image = pygame.transform.scale(image, (ASTEROID_SCALE, ASTEROID_SCALE))
	hitbox = max(image.get_width(), image.get_height())
	ASTEROID_HITBOXES[asteroid_type] = hitbox
	ASTEROID_PRELOADED[asteroid_type] = image
 
for boost in ['attack', 'defense', 'health', 'coins']:
	image = pygame.image.load(f'{PICKUP_PATH}{boost}.png')
	image = pygame.transform.scale(image, (BULLET_SCALE, BULLET_SCALE))
	hitbox = max(image.get_width(), image.get_height())
	PICKUP_HITBOXES[boost] = hitbox
	PICKUP_PRELOADED[boost] = image


BULLET_HITBOX = pygame.image.load(BULLET_PATH)
BULLET_HITBOX = pygame.transform.scale(BULLET_HITBOX, (BULLET_SCALE, BULLET_SCALE))
BULLET_PRELOADED = BULLET_HITBOX
BULLET_PRELOADED_AGENT = {}
BULLET_HITBOX = max(BULLET_HITBOX.get_width(), BULLET_HITBOX.get_height())
