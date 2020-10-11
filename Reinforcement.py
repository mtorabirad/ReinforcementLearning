import numpy as np  
from PIL import Image # To visualize the enviroment
import cv2  # for displaying
import matplotlib.pyplot as plt  
import pickle  
from matplotlib import style  
import time  

style.use("ggplot")  

SIZE = 10

NUM_EPISODES = 25000
MOVE_PENALTY = 1  
ENEMY_PENALTY = 300  
FOOD_REWARD = 25  
epsilon = 0.5  # to inject some randomness
EPS_DECAY = 0.9999  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 1000  
Max_MOVE_IN_EPI = 200

start_q_table = None 

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

# the dict! Using just for colors
d = {1: (255, 175, 0),  # blueish color
     2: (0, 255, 0),  # green
     3: (0, 0, 255)}  # red
    
class Blob:
    # A Blob is a group of connected pixels in an image that share some common property ( E.g grayscale value ). 
    # Observation space is the relative position of the food and enemy relative to the player.

    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
    
    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)
    
    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
    
    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2) # This means either -1, 0, or 1
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2) # This means either -1, 0, or 1
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1
    
########
player = Blob()
food = Blob()
enemy = Blob()
#### Save the picture of the initial position
env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
img = Image.fromarray(env, 'RGB')
img = img.resize((200, 200))  
cv2.imwrite('InitialConfig.png', np.array(img))
####

player.move()
player.action(2)

# If we already have a pre-trained q-table we will load it; otherwise, we will have to make one.
if start_q_table is None:
    # initialize the q-table#
    # The observation space will be tuple of tuples: ((x1,y1),(x2,y2))
    # The first tuple will contain the relative position of the player to the food and 
    # The second tuple will contain the relative position of the player to the enemy.
    # The keys of the q_table dictionary will be a tuple of two tuples.
    # Each observation will need four values: one value for each action.
    # So, you need four for loops.
    q_table = {}
    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            for iii in range(-SIZE+1, SIZE):
                    for iiii in range(-SIZE+1, SIZE):
                        q_table[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

print(q_table[((-9, -2), (3, 9))])

episode_rewards = [] # A list containing the final reward at the end of different episodes.

for episode in range(NUM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0:
        # Print the current episode number
        print(f"on #{episode}, epsilon is {epsilon}")
        # Average the final reward for all the episodes between the current one and the last one shown and display the average
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    
    # Start each episode with reward zero
    episode_reward = 0

    # Start iterations for the current episode
    for i in range(Max_MOVE_IN_EPI):
        obs = (player-food, player-enemy)
        #print(obs)
        if np.random.random() > epsilon:
            # GET a regular action
            action = np.argmax(q_table[obs])
        else:
            # GET a random action
            action = np.random.randint(0, 4)
        
        # Take the action!
        player.action(action)
        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############
        if player.x == enemy.x and player.y == enemy.y:
            # The player hit the enemy
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            # The player got the food
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
        
        # The player has moved. Make the new observation.
        new_obs = (player-food, player-enemy)
        
        # Use the q-table to grab the maximum q (you can expect) for a future action, which is not known yet.
        max_future_q = np.max(q_table[new_obs])

        # The crrent q is the one with which you made the most recent action while you were still at the old observation.
        current_q = q_table[obs][action]  # current Q for our chosen action

        if reward == FOOD_REWARD:
            # Once you achieved the food you are done.
            new_q = FOOD_REWARD
        else:
            # Bellman equation (simplified)
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        # Update the q-table.
        q_table[obs][action] = new_q

        if show:
            # blue: player, red: enemy, green: food
            # If you see blue and red, it means the player got the food.
            # If you see green and red, it means the player ran into the enemy. 
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
            env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red

            img = Image.fromarray(env, 'RGB')
            img = img.resize((500, 500))  
            cv2.imshow("image", np.array(img))  # show it!
            # If the player got the food or hit the enemy, pause for 500 seconds
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            # End the current episode
            break
    episode_rewards.append(episode_reward)

    # Decay the epsilor
    epsilon *= EPS_DECAY

# use numpy convolve command to create a ROLLING moving average
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7.5,7.5])
plt.plot([i for i in range(SHOW_EVERY, SHOW_EVERY+len(moving_avg))], moving_avg, c='b')
plt.scatter([i for i in range(len(episode_rewards))], episode_rewards, s=2)
plt.hlines(-MOVE_PENALTY*Max_MOVE_IN_EPI, xmin=0, xmax=NUM_EPISODES, color='k', 
            label='Made all the moves without hitting the enemy or finding the food', linewidth=4)
plt.hlines(-MOVE_PENALTY*Max_MOVE_IN_EPI-ENEMY_PENALTY, xmin=0, xmax=NUM_EPISODES, color='r', 
            label='Made all the moves and hit the enemy at the end!', linewidth=4)
plt.hlines(FOOD_REWARD, xmin=0, xmax=NUM_EPISODES, color='g', label='Found the food in the first move!', linewidth=4)

plt.legend()
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.savefig('Score')
plt.show()
print('len(moving_avg) = ', len(moving_avg))
print('len(episode_rewards) = ', len(episode_rewards))


with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)

######