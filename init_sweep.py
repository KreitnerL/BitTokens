import argparse

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="STEM", help="Wandb project name")
    parser.add_argument("--name", type=str, default="hyperparameter-sweep", help="Wandb sweep name")
    args = parser.parse_args()

    sweep_configuration = {
        'method': 'grid',  # or 'grid', 'bayes'
        'name': args.name,
        'metric': {"goal": "maximize", "name": "val_gen_acc_total"},
        'parameters': {
            'combine_strategy': {
                'values': ["sum", "prod", "concat", "weighted", "weighted_sum", "zero_pad"]
            },
            # Add other parameters you want to sweep
        }
    }

    # Initialize wandb
    wandb.login()

    # Create the sweep
    sweep_id = wandb.sweep(sweep_configuration, project=args.project)
    
    print(f"Sweep created with ID: {sweep_id}")
    print("Use this ID in your Slurm sweep agent script")
    possible_configs = 1
    for param_options in sweep_configuration['parameters'].values():
        possible_configs *= len(param_options['values'])

    print(f"There is a total of {possible_configs} possible configurations")
    print(f"\nTo start sweep agents, run:\nbash slurm_scripts/submit_sweep_agents.sh {sweep_id} {possible_configs} <config_file>")

if __name__ == "__main__":
    main()