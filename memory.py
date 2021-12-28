import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.current_device())

# Decommenting the below will give you a sense of what the allocated and reserved
# space mean.
# I just put the with statement to show that x remains on the GPU even after the with statement
# with torch.no_grad():
#     x = torch.randn(5000, 5000, dtype=torch.float64).to(device)

# Seeing total, reserved and allocated space in 0, in bytes
t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved
print(f"\nTotal memory of device 0: {t}")
print(f"memory reserved in device 0: {r}")
print(f"memory allocated in device 0: {a}")
print(f"memory free inside reservec in device 0: {f}")

# Seeing total, reserved and allocated space in 1, in bytes
t = torch.cuda.get_device_properties(1).total_memory
r = torch.cuda.memory_reserved(1)
a = torch.cuda.memory_allocated(1)
f = r-a  # free inside reserved
print(f"\nTotal memory of device 1: {t}")
print(f"memory reserved in device 1: {r}")
print(f"memory allocated in device 1: {a}")
print(f"memory free inside reservec in device 1: {f}")

print(f"memory allocated in device 0: {torch.cuda.memory_allocated(0)}")