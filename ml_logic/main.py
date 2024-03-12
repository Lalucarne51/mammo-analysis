from model import initialize_and_compile_model, train_model, optimizer, lossfn, callbacks_list
from data_flow import create_dataset, batch_dataset, train_test_split
from registry import save_model, save_results
from params import *


print("==== Starting Workflow ====")

# Step 1: Create the Dataset
print("\n=== Step 1: Creating the Dataset ===")
dataset = create_dataset(input='cloud')
print("Dataset created successfully.")
print(dataset)

# Step 2: Initialize and Compile the Model
print("\n=== Step 2: Initializing and Compiling the Model ===")
model = initialize_and_compile_model(optimizer, lossfn)
print("Model initialized and compiled successfully.")

# Step 3: Batch the Dataset
print("\n=== Step 3: Batching the Dataset ===")
batched_dataset = batch_dataset(dataset, BATCH_SIZE)
print(f"Dataset batched with batch size {BATCH_SIZE}.")

# Step 4: Create Train/Test Split
print("\n=== Step 4: Creating Train/Test Split ===")
train, test = train_test_split(batched_dataset, 0.8)
print(
    f"Train/Test split created. Train size: {len(train)}, Test size: {len(test)}."
)

# Step 5: Train the Model
print("\n=== Step 5: Training the Model ===")
history = train_model(model, train, test, epochs=500, callbacks=callbacks_list)
print("Model training complete.")

val_recall = np.min(history.history['val_recall'])
val_precision = np.min(history.history['val_precision'])
val_accuracy = np.min(history.history['val_accuracy'])

# Step 6: Save the Model
print("\n=== Step 6: Saving the Model ===")
save_model(model)
print("Model saved.")

# Step 7: Save the metrics
print("\n=== Step 7: Saving the metrics ===")
save_results(metrics=dict(recall=val_recall, precision=val_precision, accuracy=val_accuracy))
print("Metrics saved.")

# Conclusion
print("\n==== Workflow Completed Successfully ====")
