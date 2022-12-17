from numba import cuda
from numba.cuda.cudadrv import enums
print(cuda.detect())
print(cuda.devices)
print(cuda.gpus)
device = cuda.select_device(0)
print(cuda.current_context().get_memory_info())
my_sms = getattr(device, 'MULTIPROCESSOR_COUNT')
my_cc = device.compute_capability
cores_per_sm = 64 # for compute capability 7.5
total_cores = cores_per_sm*my_sms
print("GPU compute capability: " , my_cc)
print("GPU total number of SMs: " , my_sms)
print("total cores: " , total_cores)

print()
print()
print()
attribs = [s for s in dir(device) if s.isupper()]
for attr in attribs:
    print(attr, '=', getattr(device, attr))
