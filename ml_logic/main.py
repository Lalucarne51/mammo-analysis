from matplotlib import pyplot as plt
import numpy as np
import colorama
from colorama import Fore
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from model import (
    initialize_and_compile_model,
    train_model,
    optimizer,
    lossfn,
    callbacks_list,
    get_metrics,
)
from data_flow import create_dataset, batch_dataset
from registry import save_model, save_results
from params import *

colorama.init(autoreset=True)

print("==== Starting Workflow ====")

# Step 1: Create the Dataset
# all custom over_sample under_sample
print(Fore.BLUE + "\n=== Step 1: Creating the Datasets ===")
train_dataset, test_dataset = create_dataset(input="cloud", data_type="under_sample")
print(
    Fore.GREEN
    + f"Train/Test datasets created. \nTrain size: {len(train_dataset)}, Test size: {len(test_dataset)}."
)

# Step 2: Initialize and Compile the Model
print(Fore.BLUE + "\n=== Step 2: Initializing and Compiling the Model ===")
model = initialize_and_compile_model(optimizer, lossfn)
print(Fore.GREEN + "Model initialized and compiled successfully.")

# Step 3: Batch the Dataset
print(Fore.BLUE + "\n=== Step 3: Batching the Dataset ===")
batched_train_dataset = batch_dataset(train_dataset, BATCH_SIZE)
batched_test_dataset = batch_dataset(test_dataset, BATCH_SIZE)
print(Fore.GREEN + f"Dataset batched with batch size {BATCH_SIZE}.")

# Step 4: Train the Model
print(Fore.BLUE + "\n=== Step 4: Training the Model ===")
history = train_model(
    model,
    batched_train_dataset,
    batched_test_dataset,
    epochs=EPOCHS,
    callbacks=callbacks_list,
)
print(Fore.GREEN + "Model training complete.")

val_accuracy, val_precision, val_recall, val_loss = get_metrics(history)


# batched_train_dataset = batch_dataset(train, BATCH_SIZE)
# batched_test_dataset = batch_dataset(test, BATCH_SIZE)
history = train_model(
    model,
    batched_train_dataset,
    batched_test_dataset,
    epochs=EPOCHS,
    callbacks=callbacks_list,
)

    # Step 5: Obtain predicted labels
y_pred = model.predict(batched_test_dataset)
y_pred = np.where(y_pred > 0.5, 1, 0)  # Assuming binary classification

y_true = np.concatenate([y for x, y in batched_test_dataset], axis=0)# Step 6: Obtain true labels

conf_matrix = confusion_matrix(y_true, y_pred)# Step 7: Calculate confusion matrix

# Step 6: Plot confusion matrix
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Class 0", "Class 1"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Step 7: Save the Model
print(Fore.BLUE + "\n=== Step 5: Saving the Model ===")
save_model(model)
print(Fore.GREEN + "Model saved successfully.")

# Step 8: Save the metrics
print(Fore.BLUE + "\n=== Step 6: Saving the metrics ===")
save_results(
    metrics=dict(
        recall=val_recall, precision=val_precision, accuracy=val_accuracy, loss=val_loss
    )
)
print(Fore.GREEN + "Metrics saved.")

# Conclusion
print(Fore.GREEN + "\n==== Workflow Completed Successfully ====")
