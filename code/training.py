import numpy as np
from automate_training import model_training     
from utils import load_data, merge_data, DATA_PATH 
import sys

# Algorithm settings  
NUM_CELLS = {"AP": 32, "SP": 32, "TSNE_SP": 32, "TSNE_AP_SP": 32, "AP_SP": 32} 
KERNEL_SIZE = {"AP": 4, "SP": 4, "TSNE_SP": 4, "TSNE_AP_SP": 4, "AP_SP": 4} 
EPOCHS = 70
EPOCHS = 7
offset = 1

SA_data = 'data_SA.csv'
properties = np.ones(95)
properties[0] = 0
masking_value = 2
model_name = "AP_SP"

if len(sys.argv) < 2:
    print("No model selected, using", model_name, "model")
if len(sys.argv) >= 2 and sys.argv[1] not in ["AP", "SP", "TSNE_SP", "TSNE_AP_SP", "AP_SP"]:
    print("Model", sys.argv[1], "does not exist, using", model_name, "model")
if len(sys.argv) >= 2 and sys.argv[1] in ["AP", "SP", "TSNE_SP", "TSNE_AP_SP", "AP_SP"]:
    model_name = sys.argv[1]

SA, NSA = load_data(model_name, SA_data, offset, properties, masking_value)

# Calculate weight factor for NSA peptides.
# In our data, there are more peptides that do exhibit self assembly property than are those that do not.  
# During model training, we must adjust weight factors to combat this data imbalance.
factor_NSA = len(SA) / len(NSA)
 
# Merge SA nad NSA data the train and validation subsets.
all_data, all_labels = merge_data(SA, NSA)
    
# Train the model.
model_training(model_name, all_data, all_labels, NUM_CELLS[model_name], KERNEL_SIZE[model_name], EPOCHS, factor_NSA, mask_value = masking_value)