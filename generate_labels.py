import os

# Path to your Train folder
train_dir = r"C:\Users\ADITI\projects\Fruit_Ripeness\Train"

# Get class names (folder names inside Train)
class_names = sorted(os.listdir(train_dir))
class_names = [name for name in class_names if os.path.isdir(os.path.join(train_dir, name))]

# Save to labels.txt
with open("models/labels.txt", "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print(f"✅ Created labels.txt with {len(class_names)} classes:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")
