import os
import neat
import visualize
import gymnasium as gym

# Do one run with the provided genome
def run_genome(genome, config, render_mode = None):

    # Create neural network from genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Create environment
    env = gym.make("LunarLander-v2", render_mode=render_mode)
    observation, info = env.reset(seed=42)

    # Run simulation and calculate fitness
    total_reward = 0
    for _ in range(1000):
        action = net.activate(observation)
        action = action.index(max(action)) 
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            env.close()
            break
    return total_reward

# Evaluate fitness of provided genomes
def eval_genomes(genomes, config, render_mode = 'none'):
    for genome_id, genome in genomes:
        genome.fitness = run_genome(genome, config)

# Run evolution
def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Vizualize neural net
    node_names = {-1: 'input 1', -2: 'input 2', -3: 'input 3', -4: 'input 4', -5: 'input 5', -6:'input 6', -7:'input 7', -8:'input 8', 0: 'output 1', 1: 'output 2', 2: 'output 3', 3: 'output 4'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    # Vizualize stats about species and training histogram
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)

    # Run simulation with winning genome
    run_genome(winner, config, "human")
    while (True):
        reply = input("Replay winning solution? (Y/N)")
        if (reply == 'N'):
            break
        else:
            run_genome(winner, config, "human")

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config')
    run(config_path)
