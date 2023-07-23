import re
import matplotlib.pyplot as plt

# initialize lists to store data
total_memories = []
pruned_memories = []
average_pruned_memory_life = []
fraction_memories_retained_sigma = []
sigma = None

# read log file
aggregate = False
with open('logs/train_datastore_backup.log', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if '# total memories' in line:
            # extract pruned memories
            pruned = re.search(r'pruned: (\d+)', line)
            pruned_memories.append(int(pruned.group(1)))

            # extract average pruned memory life
            avg_life = re.search(r'life: (\d+\.\d+)', line)
            average_pruned_memory_life.append(float(avg_life.group(1)))

        elif 'Cache elements integrated' in line:
            # extract total memories
            total = re.search(r' new keys shape: torch.Size\(\[(\d+)', line)
            total_memories.append(int(total.group(1)))

        elif 'Using sigma=' in line:
            # extract sigma
            sigma_ = re.search(r'sigma=([+-]\d+.\d+)', line)
            sigma_ = float(sigma_.group(1))
            if sigma is None:
                sigma = sigma_
            assert sigma_ == sigma, f"Sigma is not consistent throughout the dataset ({sigma} != {sigma_})"

            # extract available memories
            available_mem = re.search(r'Available memories: (\d+)', line)
            available_mem = int(available_mem.group(1))

            # extract retained memories
            retained_mem = re.search(r'Retained memories: (\d+)', line)
            retained_mem = int(retained_mem.group(1))

            if aggregate:  # alternate indices
                available_mem_prev, retained_mem_prev = fraction_memories_retained_sigma[-1]
                fraction_mem_retained = float(retained_mem + retained_mem_prev) / (available_mem + available_mem_prev)
                fraction_memories_retained_sigma[-1] = fraction_mem_retained
            else:
                fraction_memories_retained_sigma.append((available_mem, retained_mem))  # just add the values to the list
            aggregate = ~aggregate  # flip aggregate

if not isinstance(fraction_memories_retained_sigma[-1], float):
    del fraction_memories_retained_sigma[-1]

# plot data
# update labels and figure size
fig, axs = plt.subplots(4, sharex=True, figsize=(10, 20))
plt.rcParams['font.size'] = '14'

# total memories
axs[0].plot(total_memories, label='Total Memories', color='navy')
axs[0].legend(loc='upper left')
axs[0].set_ylabel('Total Memories Count', fontsize=14)

# pruned memories
axs[1].plot(pruned_memories, label='Pruned Memories', color='maroon')
axs[1].legend(loc='upper left')
axs[1].set_ylabel('Pruned Memories Count', fontsize=14)

# average pruned memory life
axs[2].plot(average_pruned_memory_life, label='Average Pruned Memory Life', color='teal')
axs[2].legend(loc='upper left')
axs[2].set_ylabel('Average Pruned Memory Life', fontsize=14)

# fraction of memories retained with sigma threshold
label = f'Memories retained with sigma={sigma:.1f}'
axs[3].plot(fraction_memories_retained_sigma, label=label, color='green')
axs[3].legend(loc='upper left')
axs[3].set_ylabel(label, fontsize=14)

# xlabel for the last subplot
axs[2].set_xlabel('Step', fontsize=14)

plt.tight_layout()
plt.savefig("output.png", dpi=300)
plt.show()
