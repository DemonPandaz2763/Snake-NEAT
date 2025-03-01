# Snake AI Project
## Overview

This project is a snake game that features an integrated AI powered by the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. You can play the game manually or watch the AI evolve its gameplay. The project also visualizes the underlying neural network as it learns to navigate the game environment.

## Inspiration

My inspiration for the project was a problem I observed while watching Code Bullet attempt a similar project using Q Learning instead of NEAT. It was fascinating to see the neural network evolve and improve over time, though the training would eventually stagnate instead of consistently completing the game. This issue appears to be well known, with few variations ever achieving full completion. Code Bullet eventually switched to a path-finding algorithm called A*, which was disappointing since I believe that neuro-evolution represents a more authentic form of AI.

## Features

- **Manual Gameplay:** Control the snake using the arrow keys.
- **AI Training Modes:** Choose to watch the AI train in real time or run fast training simulations.
- **Neural Network Visualization:** See the evolving structure of the AI's network.
- **Neural Network Input Features:** The AI processes a 32-dimensional input vector extracted from the game state, which includes:
  - **Wall Proximity:** 8 inputs representing the normalized distance to the walls in eight directions.
  - **Food Detection:** 8 inputs indicating the normalized distance to food in each of those eight directions.
  - **Body Proximity:** 8 inputs measuring the normalized distance to the snake’s body segments in eight directions.
  - **Directional Information:** 4 inputs for the current movement direction.
  - **Tail Direction:** 4 inputs representing the tail’s relative position.
- **Automatic Checkpointing:** The best performing genome is saved and updated during training.
- **Simple Interface:** Built using Pygame for an engaging user experience.


## Requirements

 - Python 3.x
 - Pygame
 - NEAT-Python

Other dependencies include standard Python libraries such as `logging`, `pickle`, and `random`.

## Installation

1. **Clone the repository:**
    ```
    git clone https://github.com/DemonPandaz2763/snake-ai-project.git
    ```
2. **Navigate to Snake files**
    ```
    cd snake-ai-project
    ```
3. **Install Requirements**
    ```
    pip3 install -r requirements.txt
    ```

## Usage

Before running the game, ensure that you have a config.txt file in the project directory. This is the AI's, it doesn't like to play without it.

Run the game using:
```
python3 main.py
```

## Game modes
 - 1: Play Snake (manual mode)
 - 2: Watch AI train over time
 - 3: Train AI Fast (much much much quicker)
 - 4: Watch Best AI (added for the faster trained AI)

 ## Note
 
 This is a first draft of the program. I have no idea if it actually completes the game or not, though that is my goal. As of now the code is very bugging and very messy. I'm working on that :)
