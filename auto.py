import subprocess
import os
import time

# Define the input file paths and algorithms
input_files = [f"Sources/Test_cases/input0{i}.txt" for i in range(1, 6)]
algorithms = ['bfs','dfs','ucs', 'greedy', 'astar', 'dls', 'ids', 'bidirect']

# Function to close the popup (example for Unix-like systems using xdotool)


def close_popup():
    try:
        subprocess.run(["xdotool", "search", "--name",
                       "Popup Window Name", "windowkill"], check=True)
    except subprocess.CalledProcessError as e:
        print("No popup window to close or error in closing window:", e)


# Iterate over each combination of input file and algorithm
for algorithm in algorithms:
    for input_file in input_files:
        command = f"python Sources/main.py {input_file} {algorithm}"
        print(f"Executing: {command}")
        subprocess.run(command, shell=True)

        # Simulate waiting for the process to complete
        time.sleep(5)  # Adjust based on expected run time of main.py

        # Close the popup screen (if applicable)
        close_popup()

        # Run `rm *.eps` to delete EPS files
        try:
            subprocess.run("rm *.eps", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print("Error in removing EPS files:", e)

        print(f"Completed: {command}\n")

print("All processes completed.")
