import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the dictionary-style JSON (columns → lists)
with open("pose_logs.json", "r") as f:
    data = json.load(f)

# Convert dictionary of lists → Pandas DataFrame
df = pd.DataFrame(data)

print("Loaded rows:", len(df))
print(df.head())

# -----------------------------------------------------
# 1️⃣ FPS Over Time
# -----------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(df["time"], df["fps"], label="FPS", color="green")
plt.xlabel("Time (Unix seconds)")
plt.ylabel("FPS")
plt.title("FPS Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("fps_over_time.png")
plt.show()

# -----------------------------------------------------
# 2️⃣ Action Timeline
# -----------------------------------------------------
df["action_id"] = df["actions"].astype("category").cat.codes

plt.figure(figsize=(12, 4))
plt.plot(df["frames"], df["action_id"], marker="o", linestyle="-")
plt.xlabel("Frame")
plt.ylabel("Action Code")
plt.title("Action Timeline")
plt.grid(True)
plt.tight_layout()
plt.savefig("action_timeline.png")
plt.show()

print("\nAction Mapping:")
print(df[["actions", "action_id"]].drop_duplicates())

# -----------------------------------------------------
# 3️⃣ Knee Angle Histogram
# -----------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(df["knee_angles"].dropna(), bins=40, color="blue", alpha=0.7)
plt.xlabel("Knee Angle (degrees)")
plt.ylabel("Frequency")
plt.title("Knee Angle Distribution")
plt.tight_layout()
plt.savefig("knee_angle_histogram.png")
plt.show()

# -----------------------------------------------------
# 4️⃣ Hip Movement Over Time
# -----------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(df["time"], df["movement"], color="red")
plt.xlabel("Time (Unix seconds)")
plt.ylabel("Hip Movement (pixels)")
plt.title("Hip Movement Velocity Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("hip_movement.png")
plt.show()
