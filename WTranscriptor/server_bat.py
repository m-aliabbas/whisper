import subprocess
import multiprocessing

def activate_conda_environment(conda_env):
    try:
        activate_command = f'source activate {conda_env}'
        subprocess.run(activate_command, shell=True, check=True, executable='/bin/bash')
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def run_server(instance_number, conda_env,port):
    try:
        # Activate the Conda environment
        activate_conda_environment(conda_env)

        # Run uvicorn with a unique port for each instance
        uvicorn_command = f'uvicorn server:app --reload --port {port}'
        subprocess.run(uvicorn_command, shell=True, check=True, executable='/bin/bash')
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Process interrupted.")

if __name__ == "__main__":
    # Specify the Conda environment name
    conda_environment = "pytorch_latest"

    # Number of server instances to run
    num_instances = 10  

    # Create a list of processes for the server instances
    processes = []
    for i in range(num_instances):
        port = 8000+i
        process = multiprocessing.Process(target=run_server, args=(i, conda_environment,port))
        processes.append(process)

    # Start the processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()
