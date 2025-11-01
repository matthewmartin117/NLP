import torch
import time
import pandas as pd

# Check if GPU is available
device_cuda = torch.device('cuda' )  # careful, if you don't have a GPU and CUDA installed, this will cause problems down the line
device_cpu = torch.device('cpu')
#print(f'Using device: {device}')

# Function to perform matrix multiplication and measure time
def measure_time(n, device):
    # Create two random matrices of size n x n
    A = torch.rand((n, n), device=device)
    B = torch.rand((n, n), device=device)
    
    # Record the start time
    start_time = time.time()
    
    # Perform matrix multiplication
    C = torch.matmul(A, B)
    
    # Ensure the operation is complete
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Record the end time
    end_time = time.time()
    
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    return elapsed_time

# Define matrix sizes and step size
sizes = [10, 100, 500, 1000, 2500, 5000, 7500, 10000, 20000]

# DataFrame to store results
results = []

# Measure and record time for each matrix size
print("N\tCPU\tGPU")
for n in sizes:
    elapsed_time_cpu = measure_time(n, device_cpu)
    elapsed_time_gpu = measure_time(n, device_cuda)
    results.append({f'{n}\t{elapsed_time_cpu:.4f}\t{elapsed_time_gpu:.4f}'})
    print(f'{n}\t{elapsed_time_cpu:.4f}\t{elapsed_time_gpu:.4f}')
    #print(f'Matrix Size: {n} x {n}, Time: {elapsed_time:.6f} seconds')

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# # Display the results
# print('\nPerformance Comparison:')
# print(results_df)

