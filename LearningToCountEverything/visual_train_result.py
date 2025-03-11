import matplotlib.pyplot as plt
import re

#metrics
epochs = []
avg_epoch_loss = []
train_mae = []
train_rmse = []
val_mae = []
val_rmse = []

result_path='result/stats.txt'
with open(result_path) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        numbers = re.findall(r"[-+]?\d*\.\d+e[-+]\d+|[-+]?\d*\.\d+|[-+]?\d+", line)
        numbers = [float(num) for num in numbers]

        # [epo loss, train mae, train rmse, val mae, val rmse]
        epochs.append(len(epochs) + 1)
        avg_epoch_loss.append(numbers[0])
        train_mae.append(numbers[1])
        train_rmse.append(numbers[2])
        val_mae.append(numbers[3])
        val_rmse.append(numbers[4])


# tạo figure
plt.figure(figsize=(12, 8))

# Plot for Loss
plt.subplot(2, 2, 1)
plt.plot(epochs, avg_epoch_loss, marker="", color="blue")
plt.title("Avg. Epoch Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# Plot for MAE
plt.subplot(2, 2, 2)
plt.plot(epochs, train_mae, marker="", label="Train MAE")
plt.plot(epochs, val_mae, marker="", label="Val MAE")
plt.title("MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)

# Plot for RMSE
plt.subplot(2, 2, 3)
plt.plot(epochs, train_rmse, marker="", label="Train RMSE")
plt.plot(epochs, val_rmse, marker="", label="Val RMSE")
plt.title("RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
