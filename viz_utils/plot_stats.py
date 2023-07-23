import re
import matplotlib.pyplot as plt

# initialize lists to store data
total_memories = []
pruned_memories = []
average_pruned_memory_life = []

# read log file
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

# plot data
# update labels and figure size
fig, axs = plt.subplots(3, sharex=True, figsize=(10, 15))
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

# xlabel for the last subplot
axs[2].set_xlabel('Step', fontsize=14)

plt.tight_layout()
plt.savefig("output.png", dpi=300)
plt.show()
