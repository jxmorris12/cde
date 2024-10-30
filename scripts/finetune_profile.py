from torch.profiler import profile, record_function, ProfilerActivity

from finetune import main

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    main()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print("\n\n")
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
print("\n\n")
print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))
print("\n\n")
print("exporting chrome trace to trace.json")
prof.export_chrome_trace("trace.json")
print("exporting stacks to profiler_stacks.txt")
prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")
print("done")
