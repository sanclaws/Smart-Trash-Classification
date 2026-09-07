import json
import matplotlib.pyplot as plt

with open("training_history.json", "r") as f:
    history = json.load(f)

acc = history["accuracy"]
val_acc = history["val_accuracy"]

loss = history["loss"]
val_loss = history["val_loss"]

epochs_range = range(len(acc))

plt.figure(figsize=(8,5))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig("accuracy_graph.png")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig("loss_graph.png")
plt.show()