# check_progress.py
import os, numpy as np

actions = ["hello","thank_you","I_Love_You","yes","no","help","please","cat","eat","fine"]
DATA_PATH = "MP_Data"
seq_len = 30

print("Checking sample file and counts:")
for a in actions:
    ap = os.path.join(DATA_PATH, a)
    if not os.path.exists(ap):
        print(f"{a}: MISSING")
        continue
    seqs = [d for d in os.listdir(ap) if d.isdigit()]
    print(f"{a}: {len(seqs)} sequences")
    if len(seqs) > 0:
        sample = np.load(os.path.join(ap, seqs[0], "0.npy"))
        print("  sample shape:", sample.shape)
print("Done.")
