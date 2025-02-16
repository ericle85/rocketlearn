import subprocess
import sys
from tqdm.rich import tqdm
import time
import argparse

def launch_workers(num_workers):
    processes = []
    try:
        print("Initializing workers..")
        for _ in tqdm(range(num_workers)):
            p = subprocess.Popen(["python", "-m", "examples.default.worker"])
            processes.append(p)
            time.sleep(0.2)
        
        for p in processes:
            p.wait()

    except KeyboardInterrupt:
        for p in processes:
            try:
                p.terminate()
                p.wait()
            except Exception as e:
                print(f"Failed to terminate process {p.pid}: {e}")
    finally:
        sys.exit(0)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Launch multiple worker processes.")

    # Add num_workers argument with default value of 16
    parser.add_argument(
        "--num_workers",
        type=int,
        default=15,
        help="Number of worker processes to launch (default: 16)"
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Launch workers with the specified or default number of workers
    launch_workers(args.num_workers)
    # For me, optimal is around 28 with 64vcpus, getting 46ksps
