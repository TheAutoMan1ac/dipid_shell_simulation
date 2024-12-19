import subprocess
import time
import argparse
import yaml

def run_simulation_subprocess(alpha_sticky_deg_value, n_steps, save_every_batch, plot_every_batch, random_placement, random_chance, add_unit_every, batch_mode=True):
    # Build the command to run your simulation script with arguments
    command = [
        'python', 'simulation_assembly.py',  # Replace with your actual script name,
        '--alpha_sticky_deg', str(alpha_sticky_deg_value),
        '--n_steps', str(n_steps),
        '--save_every_batch', str(save_every_batch),
        '--plot_every_batch', str(plot_every_batch),
        '--random_chance', str(random_chance),
        '--add_unit_every', str(add_unit_every)
    ]

    if random_placement:
        command.append('--random_placement')

    if batch_mode:
        command.append('--batch_mode')

    process = subprocess.Popen(command)
    return process

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run multiple simulations based on configuration file.')
    parser.add_argument('--config_file', type=str, default='simulation_config.yaml', help='Path to the configuration file')
    parser.add_argument('--n_steps', type=int, default=10000000000, help='Number of steps for each simulation (default: 10000000)')
    parser.add_argument('--max_concurrent', type=int, default=20, help='Maximum number of concurrent simulations (default: 5)')
    parser.add_argument('--save_every_batch', type=int, default=50, help='Frequency to save simulation (default: 20)')
    parser.add_argument('--plot_every_batch', type=int, default=100, help='Frequency to plot simulation (default: 20)')
    parser.add_argument('--add_unit_every', type=int, default=20000, help="Chance of randomly placing a monomer")
    args = parser.parse_args()

    config_file = args.config_file
    n_steps = args.n_steps
    max_concurrent_processes = args.max_concurrent
    save_every_batch = args.save_every_batch
    plot_every_batch = args.plot_every_batch
    add_unit_every = args.add_unit_every

    # Read the configuration file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    simulations = config.get('simulations', [])

    processes = []
    for sim_entry in simulations:
        random_placement = sim_entry.get('random_placement', False)
        random_chance = sim_entry.get('random_chance', 0.0)
        repeats = sim_entry.get('repeats', 1)
        alpha_values = sim_entry.get('alpha_sticky_deg_values', [])

        for alpha_sticky_deg in alpha_values:
            for i in range(repeats):
                # Start the subprocess
                process = run_simulation_subprocess(
                    alpha_sticky_deg,
                    n_steps,
                    save_every_batch,
                    plot_every_batch,
                    random_placement,
                    random_chance,
                    add_unit_every,
                    batch_mode=True
                )
                processes.append(process)

                # Check if we've reached the maximum number of concurrent processes
                while len(processes) >= max_concurrent_processes:
                    # Poll processes to see if any have finished
                    for p in processes:
                        if p.poll() is not None:
                            # Process has finished
                            processes.remove(p)
                            break
                    else:
                        # No processes have finished yet; wait a bit before checking again
                        time.sleep(10)

    # Wait for any remaining processes to finish
    for p in processes:
        p.wait()
