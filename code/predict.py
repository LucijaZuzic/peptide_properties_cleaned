import numpy as np
import sys  
from utils import load_data, merge_data, reshape_for_model, convert_list, MAX_LEN 
from automate_training import BATCH_SIZE
import tensorflow as tf

if len(sys.argv) > 1 and len(sys.argv[1]) <= MAX_LEN:
    pep_list = [sys.argv[1]] 
    model = "AP_SP"
    if len(sys.argv) > 2 and sys.argv[2] in ["AP", "SP", "TSNE_SP", "TSNE_AP_SP", "AP_SP"]:
        model = sys.argv[2] 
    else:
        if len(sys.argv) < 3:
            print("No model selected, using", model, "model")
        if len(sys.argv) >= 3 and sys.argv[2] not in ["AP", "SP", "TSNE_SP", "TSNE_AP_SP", "AP_SP"]:
            print("Model", sys.argv[2], "does not exist, using", model, "model")
        
    seq_example = ""
    for i in range(MAX_LEN):
        seq_example += "A"
    pep_list.append(seq_example)
    pep_labels = ["1", "1"]

    best_batch_size = 600
    best_model = ""
    NUM_TESTS = 5

    offset = 1

    properties = np.ones(95)
    properties[0] = 0
    masking_value = 2
  
    SA, NSA = load_data(model, [pep_list, pep_labels], offset, properties, masking_value)
    all_data, all_labels = merge_data(SA, NSA) 
    
    # Load the best model.
    best_model = tf.keras.models.load_model("../models/old/" + model + ".h5")
 
    # Get model predictions on the test data.
    test_data, test_labels = reshape_for_model(model, all_data, all_labels)
    model_predictions = best_model.predict(test_data, batch_size = BATCH_SIZE)
    model_predictions = convert_list(model_predictions)  

    print(model_predictions[0])
else:
    if len(sys.argv) <= 1:
        print("No peptide")
    if len(sys.argv[0]) > MAX_LEN:
        print("Peptide too long")