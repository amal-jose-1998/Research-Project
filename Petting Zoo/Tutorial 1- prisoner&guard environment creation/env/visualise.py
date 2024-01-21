import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from custom_environment import CustomEnvironment  
import numpy as np

def visualize_environment(env, num_steps=100):
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        # Visualize colored tiles
        grid = env.render()  # Get the grid from the environment's render method
        ax.imshow(grid == "P", cmap='Blues', vmin=0, vmax=1, alpha=0.5) 
        ax.imshow(grid == "G", cmap='Reds', vmin=0, vmax=1, alpha=0.5)
        ax.imshow(grid == "E", cmap='Greens', vmin=0, vmax=1, alpha=0.5)
        # Display labels over colored tiles
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] == "P":
                    ax.text(x, y, "Prisoner", ha='center', va='center', color='black', fontsize=8, fontweight='bold')
                elif grid[y, x] == "G":
                    ax.text(x, y, "Guard", ha='center', va='center', color='black', fontsize=8, fontweight='bold')
                elif grid[y, x] == "E":
                    ax.text(x, y, "Escape", ha='center', va='center', color='black', fontsize=8, fontweight='bold')

        if env.agents:  # Check if agents are present
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}           
            obs, rewards, terminations, truncations, infos = env.step(actions)
            if any(terminations.values()) or all(truncations.values()):
                anim.event_source.stop()
            print(rewards)
            print(obs)

    anim = FuncAnimation(fig, update, frames=num_steps, interval=500, repeat=False)
    plt.show()

if __name__ == "__main__":
    env = CustomEnvironment()   
    for episode in range(10):
        obs, infos = env.reset()
        visualize_environment(env, num_steps=50)  # Adjust the number of steps as needed
      
