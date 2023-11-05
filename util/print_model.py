import argparse

def print_model(args):
    if args.continuous == 'True':
        mode = "continuous"
    else:
        mode = "discrete"

    if args.cql == 'True':
        cql = "+cql"
    else:
        cql = "-cql"

    print(f"mode: {mode}||"
        f"cql: {cql}||"
        f"state_dim: {args.state_size}||"
        f"action_dim: {args.action_size}||"
        f"learning_starts: {args.learning_starts}||"
        f"total_steps: {args.total_timesteps}||"
        f"rank_update_interval: {args.rank_update_interval}||"
        f"batch_size: {args.batch_size}||"
        f"device: {args.device}")
