import re
import matplotlib.pyplot as plt


# initialize lists to store data
total_memories = []
pruned_memories = []
average_pruned_memory_life = []
fraction_memories_retained_sigma = []
nn_distance = []
nn_memory_life = []
sigma = None

# read log file
aggregate = False
with open('logs/train_shuffled_datastore_adaptive_lambda_main_beta_0.33.log', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if '# total memories' in line:
            # extract pruned memories
            pruned = re.search(r'pruned: (\d+)', line)
            pruned_memories.append(int(pruned.group(1)))

            # extract average pruned memory life
            avg_life = re.search(r'life: (\d+\.\d+)', line)
            average_pruned_memory_life.append(float(avg_life.group(1)))

        elif 'Retrieved nearest neighbors' in line:
            # Retrieved nearest neighbors / Distance: (min: 0.38, mean: 22.55, max: 43.20) / Nearest neighbors memory life: (min: 0.00, mean: 6033.44, max: 33598.00)
            nn_dist = re.search(r'Distance: \(min: (\d+\.\d+), mean: (\d+\.\d+), max: (\d+\.\d+)\)', line)
            nn_mem_life = re.search(r'Nearest neighbors memory life: \(min: (\d+\.\d+), mean: (\d+\.\d+), max: (\d+\.\d+)\)', line)
            nn_distance.append((float(nn_dist.group(1)), float(nn_dist.group(2)), float(nn_dist.group(3))))
            nn_memory_life.append((float(nn_mem_life.group(1)), float(nn_mem_life.group(2)), float(nn_mem_life.group(3))))

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
num_plots = 6
fontsize = 14
fig, axs = plt.subplots(num_plots, sharex=True, figsize=(10, num_plots*5))
plt.rcParams['font.size'] = str(fontsize)

# total memories
axs[0].plot(total_memories, label='Total Memories', color='navy')
axs[0].legend(loc='upper left')
axs[0].set_ylabel('Total Memories Count', fontsize=fontsize)

# pruned memories
axs[1].plot(pruned_memories, label='Pruned Memories', color='maroon')
axs[1].legend(loc='upper left')
axs[1].set_ylabel('Pruned Memories Count', fontsize=fontsize)

# average pruned memory life
axs[2].plot(average_pruned_memory_life, label='Average Pruned Memory Life', color='teal')
axs[2].legend(loc='upper left')
axs[2].set_ylabel('Average Pruned Memory Life', fontsize=fontsize)

# fraction of memories retained with sigma threshold
label = f'Memories retained with sigma={sigma:.1f}'
axs[3].plot(fraction_memories_retained_sigma, label=label, color='green')
axs[3].legend(loc='upper left')
axs[3].set_ylabel(label, fontsize=fontsize)

# nn distance
axs[4].plot([x[0] for x in nn_distance], label='min')
axs[4].plot([x[1] for x in nn_distance], label='mean')
axs[4].plot([x[2] for x in nn_distance], label='max')
axs[4].legend(loc='upper left')
axs[4].set_ylabel('Nearest neighbor distance', fontsize=fontsize)

# nn memory life
axs[5].plot([x[0] for x in nn_memory_life], label='min')
axs[5].plot([x[1] for x in nn_memory_life], label='mean')
axs[5].plot([x[2] for x in nn_memory_life], label='max')
axs[5].legend(loc='upper left')
axs[5].set_ylabel('Nearest neighbor memory life', fontsize=fontsize)

# xlabel for the last subplot
axs[-1].set_xlabel('Step', fontsize=fontsize)
for ax in axs:
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)

plt.tight_layout()
plt.savefig("output.png", dpi=300)
plt.show()
