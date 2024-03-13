import numpy as np
import colorama
from colorama import Fore
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

# Const for the test
TAKE = 100
BATCH_SIZE = 10
EPOCHS = 2

colorama.init(autoreset=True)

print("==== Starting Test Workflow ====")

# Step 1: Create the Dataset
print(Fore.BLUE + "\n=== Step 1: Creating the Datasets ===")
train_dataset, test_dataset = create_dataset(input="cloud", data_type="custom")
print(
    Fore.GREEN
    + f"Train/Test split created. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}."
)

print(Fore.BLUE + "\n=== Step 2: Creating the Small Datasets ===")
train_dataset_small = train_dataset.take(TAKE)
test_dataset_small = test_dataset.take(TAKE)
print(Fore.GREEN + "Small datasets created successfully.")

# Step 3: Initialize and Compile the Model
print(Fore.BLUE + "\n=== Step 3: Initializing and Compiling the Model ===")
model = initialize_and_compile_model(optimizer, lossfn)
print(Fore.GREEN + "Model initialized and compiled successfully.")

# Step 4: Batch the Dataset
print(Fore.BLUE + "\n=== Step 4: Batching the Dataset ===")
batched_train_dataset_small = batch_dataset(train_dataset_small, BATCH_SIZE)
batched_test_dataset_small = batch_dataset(test_dataset_small, BATCH_SIZE)
print(Fore.GREEN + f"Dataset batched with batch size {BATCH_SIZE}.")

# Step 5: Train the Model
print(Fore.BLUE + "\n=== Step 5: Training the Model ===")
history = train_model(
    model,
    batched_train_dataset_small,
    batched_test_dataset_small,
    epochs=EPOCHS,
    callbacks=callbacks_list,
)
print(Fore.GREEN + "Model training complete.")

val_accuracy, val_precision, val_recall, val_loss = get_metrics(history)

# Step 6: Save the Model
print(Fore.BLUE + "\n=== Step 6: Saving the Model ===")
save_model(model)
print(Fore.GREEN + "Model saved successfully.")

# Step 7: Save the metrics
print(Fore.BLUE + "\n=== Step 7: Saving the metrics ===")
save_results(
    metrics=dict(
        recall=val_recall, precision=val_precision, accuracy=val_accuracy, loss=val_loss
    )
)
print(Fore.GREEN + "Metrics saved.")

# Conclusion
print(Fore.GREEN + "\n==== Workflow Completed Successfully ====")
