from model import (
    initialize_and_compile_model,
    train_model,
    optimizer,
    lossfn,
    callbacks_list,
)
import numpy as np
from data_flow import create_dataset, batch_dataset
from registry import save_model, save_results
from params import *


print("==== Starting Workflow ====")

# Step 1: Create the Dataset
print("\n=== Step 1: Creating the Datasets ===")
train_dataset, test_dataset = create_dataset(input="cloud")
print("Dataset created successfully.")
print(
    f"Train/Test split created. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}."
)
print(train_dataset, test_dataset)

# Step 2: Initialize and Compile the Model
print("\n=== Step 2: Initializing and Compiling the Model ===")
model = initialize_and_compile_model(optimizer, lossfn)
print("Model initialized and compiled successfully.")

# Step 3: Batch the Dataset
print("\n=== Step 3: Batching the Dataset ===")
batched_train_dataset = batch_dataset(train_dataset, BATCH_SIZE)
batched_test_dataset = batch_dataset(test_dataset, BATCH_SIZE)
print(f"Dataset batched with batch size {BATCH_SIZE}.")


# Step 4: Train the Model
print("\n=== Step 4: Training the Model ===")
history = train_model(
    model,
    batched_train_dataset,
    batched_test_dataset,
    epochs=1,
    callbacks=callbacks_list,
)
print("Model training complete.")

val_recall = np.min(history.history["val_recall"])
val_precision = np.min(history.history["val_precision"])
val_accuracy = np.min(history.history["val_accuracy"])

# Step 5: Save the Model
print("\n=== Step 5: Saving the Model ===")
save_model(model)
print("Model saved.")

# Step 6: Save the metrics
print("\n=== Step 6: Saving the metrics ===")
save_results(
    metrics=dict(recall=val_recall, precision=val_precision, accuracy=val_accuracy)
)
print("Metrics saved.")

# Conclusion
print("\n==== Workflow Completed Successfully ====")
