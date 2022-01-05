import multiprocessing
from datetime import datetime
import time

def practice_function() -> None:
    """Function for practising my use of Python's multiprocessing library.
    """

    num_cycles = 10

    # Adding a sleep function to increase the length of time this practice function 
    # takes to run, so I can estimate if the multiprocessing method further below saves 
    # time overall with my simulations

    time.sleep(10)

    i = 0
    for cycle in range(num_cycles):
        i += 1
        print(i)

print(f"Number of CPUs available: {multiprocessing.cpu_count()}")
num_processes = 18

# A list for appending processes that allows one to have all processes be completed before the part of the script beyond 
# their initation runs
processes = []

start = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
print("Started at:", start)

# Another start variable, computed to later facilitate printing how long the processes took
start = time.perf_counter()

for process in range(num_processes):
    p = multiprocessing.Process(target=practice_function)
    p.start()
    processes.append(p)

for process in processes:
    process.join()

finish = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
print("Finished at:", finish)

finish = time.perf_counter()
print(f"Finished in {round(finish - start, 2)} seconds")

# With the practice function containg a sleep time of 10 seconds, the 
# script completed in 10.02 seconds with the multiprocessing method from 
# https://www.youtube.com/watch?v=fKl2JW_qrso&t=814s - it definitely 
# seems to work, especially since all 18 processes seem to have run.