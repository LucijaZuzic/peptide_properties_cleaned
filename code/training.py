import numpy as np
from automate_training import load_data_SA, merge_data
from utils import DATA_PATH
import matplotlib.pyplot as plt  
import sys 
import new_model
import os
import tensorflow as tf 
from automate_training import return_callbacks, LEARNING_RATE_SET, reshape

# Algorithm settings 
N_FOLDS_FIRST = 5
N_FOLDS_SECOND = 5
EPOCHS = 70
names = ['AP']
offset = 1

SA_data = np.load(DATA_PATH+'data_SA_updated.npy', allow_pickle=True).item()
  
properties = np.ones(95) 
masking_value = 2
SA, NSA = load_data_SA(SA_data, names, offset, properties, masking_value)

# Calculate weight factor for NSA peptides.
# In our data, there are more peptides that do exhibit self assembly property than are those that do not. Therefore,
# during model training, we must adjust weight factors to combat this data imbalance.
factor_NSA = len(SA) / len(NSA)
 
# Merge SA nad NSA data the train and validation subsets.
all_data, all_labels = merge_data(SA, NSA) 
num_props= len(names) * 3

model_type = -1
# Convert train and validation indices to train and validation data and train and validation labels
train_data, train_labels = reshape(num_props, all_data, all_labels)

#python program to check if a path exists
#if it doesn’t exist we create one
if not os.path.exists("../final_all/"):
    os.makedirs("../final_all/")

# Write output to file
sys.stdout = open("../final_all/training_log.txt", "w", encoding="utf-8")

# Save model to correct file based on number of fold
model_file, model_picture = "../final_all/model.h5", "../final_all/model_picture.png"

# Choose correct model and instantiate model 
model = new_model.amino_di_tri_model(num_props, input_shape=np.shape(train_data[num_props][0]), conv=5, numcells=32, kernel_size=4, lstm1=5, lstm2=5, dense=15, dropout=0.5, lambda2=0.0, mask_value=2)

# Save graphical representation of the model to a file.
tf.keras.utils.plot_model(model, to_file=model_picture, show_shapes=True)

# Print model summary.
model.summary()

callbacks = return_callbacks(model_file, 'loss') 

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_SET)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
) 

count_len_SA = 0
count_len_NSA = 0
for label in train_labels:
    if label == 1:
        count_len_SA += 1
    else:
        count_len_NSA += 1
factor_NSA = count_len_SA / count_len_NSA 

history =  model.fit(
    train_data, 
    train_labels,
    #validation_split = 0.1,  
    #validation_data=[val_data, val_labels],
    class_weight={0: factor_NSA, 1: 1.0}, 
    epochs=EPOCHS,
    batch_size = 600,
    callbacks=callbacks,
    verbose=1
) 
 
accuracy = history.history['accuracy']
loss = history.history['loss'] 

# Output accuracy, validation accuracy, loss and validation loss for all models
accuracy_max = np.max(accuracy) 
loss_min = np.min(loss) 

other_output = open("../final_all/results_accuracy_loss.txt", "w", encoding="utf-8")
other_output.write(
    "Maximum accuracy = %.12f%% Minimal loss = %.12f"
    % ( 
        accuracy_max * 100, 
        loss_min, 
    )
)
other_output.write("\n")
other_output.write(
    "Accuracy = %.12f%% (%.12f%%) Loss = %.12f (%.12f)"
    % (
        np.mean(accuracy) * 100,
        np.std(accuracy) * 100,
        np.mean(loss),
        np.std(loss),
    )
)
other_output.write("\n")
other_output.close()

other_output = open("../final_all/accuracy.txt", "w", encoding="utf-8")
other_output.write(str(accuracy))
other_output.write("\n")
other_output.close()

other_output = open("../final_all/loss.txt", "w", encoding="utf-8")
other_output.write(str(loss))
other_output.write("\n")
other_output.close()

# Plot the history

# Summarize history for accuracy
plt.figure()
plt.plot(history.history["accuracy"], label="Accuracy")
plt.title(
    "Final model seq. props. and AP\n Accuracy"
)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.savefig("../final_all/accuracy.png", bbox_inches="tight")
plt.close()
# Summarize history for loss
plt.figure()
plt.plot(history.history["loss"], label="Loss")
plt.title(
    "Final model seq. props. and AP\n Loss" 
)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.savefig("../final_all/loss.png", bbox_inches="tight")
plt.close()

# Close output file
sys.stdout.close()