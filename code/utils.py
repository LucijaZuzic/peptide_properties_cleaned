DATA_PATH = '../data/'
MODEL_DATA_PATH = '../model_data/'
SEQ_MODEL_DATA_PATH = '../seq_model_data/'
MY_MODEL_DATA_PATH = '../only_my_model_data/' 
TSNE_AP_SEQ_DATA_PATH = '../TSNE_ap_seq_model_data/' 
TSNE_SEQ_DATA_PATH = '../TSNE_seq_model_data/' 
MAXLEN = 24

from seqprops import SequentialPropertiesEncoder  
from sklearn.preprocessing import MinMaxScaler 
import numpy as np
 
def set_seed(x):
    seed_file = open(DATA_PATH + "seed.txt", "w")
    seed_file.write(str(x))
    seed_file.close()
 
def get_seed():
    seed_file = open(DATA_PATH + "seed.txt", "r")
    line = seed_file.readlines()[0].replace("\n", "")
    seed_file.close()
    return int(line)

def data_and_labels_from_indices(all_data, all_labels, indices):
    data = []
    labels = []

    for i in indices:
        data.append(all_data[i])
        labels.append(all_labels[i]) 
        
    return data, labels 

def scale(AP_dictionary, offset = 1):
    data = [AP_dictionary[key] for key in AP_dictionary]

    # Determine min and max AP scores.
    min_val = min(data)
    max_val = max(data)

    # Scale AP scores to range [- offset, offset].
    for key in AP_dictionary:
        AP_dictionary[key] = (AP_dictionary[key] - min_val) / (max_val - min_val) * 2 * offset - offset
   
def padding(array, len_to_pad, value_to_pad):

    # Fill array with padding value to maximum length.
    new_array = [value_to_pad for i in range(len_to_pad)]
    for val_index in range(len(array)):

        # Add original values.
        if val_index < len(new_array):
            new_array[val_index] = array[val_index]
    return new_array

def split_amino_acids(sequence, amino_acids_AP_scores):
    ap_list = []

    # Replace each amino acid in the sequence with a corresponding AP score.
    for letter in sequence:
        ap_list.append(amino_acids_AP_scores[letter])

    return ap_list

def split_dipeptides(sequence, dipeptides_AP_scores):
    ap_list = []

    # Replace each dipeptide in the sequence with a corresponding AP score.
    for i in range(len(sequence) - 1):
        ap_list.append(dipeptides_AP_scores[sequence[i:i + 2]])

    return ap_list

def split_tripeptides(sequence, tripeptides_AP_scores):
    ap_list = []

    # Replace each tripeptide in the sequence with a corresponding AP score.
    for i in range(len(sequence) - 2):
        ap_list.append(tripeptides_AP_scores[sequence[i:i + 3]])

    return ap_list

def reshape_for_model(model_name, num_props, all_data, all_labels):

    labels = []
    for i in range(len(all_data)):
        labels.append(all_labels[i]) 
    if len(labels) > 0:
        labels = np.array(labels)

    if "SP" in model_name and "AP" not in model_name:

        data = []
        for i in range(len(all_data)):
            data.append(np.array(all_data[i]))
        if len(data) > 0:
            data = np.array(data)
        return data, labels
     
    data = [[] for i in range(len(all_data[0]))]
    for i in range(len(all_data)):
        for j in range(len(all_data[0])):
            data[j].append(all_data[i][j])

    new_data = []
    last_data = []    
    for i in range(len(data)):
        if len(data[i]) > 0 and i < num_props:  
            new_data.append(np.array(data[i]))
        if "AP" in model_name and "SP" in model_name:
            if len(data[i]) > 0 and i >= num_props: 
                last_data.append(np.array(data[i])) 

    if len(last_data) > 0:
        last_data = np.array(last_data).transpose(1, 2, 0)
        new_data.append(last_data)
  
    if len(new_data) > 0:
        new_data = np.array(new_data)
    return new_data, labels 
   
# Choose loading AP, APH, logP or polarity
def load_data_AP(name = 'AP', offset = 1):
    # Load AP scores. 
    amino_acids_AP = np.load(DATA_PATH+'amino_acids_'+name+'.npy', allow_pickle=True).item()
    dipeptides_AP = np.load(DATA_PATH+'dipeptides_'+name+'.npy', allow_pickle=True).item()
    tripeptides_AP = np.load(DATA_PATH+'tripeptides_'+name+'.npy', allow_pickle=True).item()
    
    # Scale scores to range [-1, 1].
    scale(amino_acids_AP, offset)
    scale(dipeptides_AP, offset)
    scale(tripeptides_AP, offset)

    return amino_acids_AP, dipeptides_AP, tripeptides_AP

def read_from_npy_SA(SA_data):
    sequences = []
    labels = []
    for peptide in SA_data:
        if len(peptide) > MAXLEN or SA_data[peptide] == '-1':
            continue
        sequences.append(peptide)
        labels.append(SA_data[peptide])

    return sequences, labels

def load_data_SA_seq(SA_data, names=['AP'], offset = 1, properties_to_include = [], masking_value=2):
    sequences, labels = read_from_npy_SA(SA_data)
            
    # Encode sequences
    encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-offset, offset)))
    encoded_sequences = encoder.encode(sequences)
     
    # Split peptides in two bins.
    # SA - has self-assembly, NSA - does not have self-assembly.
    # Convert each peptide sequence to three lists of AP scores: amino acid, dipeptide and tripeptide scores.
    SA = []
    NSA = []
    for index in range(len(sequences)):
        new_props = []
        for name in names:
            amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP(name, offset)  
            amino_acids_ap = split_amino_acids(sequences[index], amino_acids_AP)
            dipeptides_ap = split_dipeptides(sequences[index], dipeptides_AP)
            tripeptides_ap = split_tripeptides(sequences[index], tripeptides_AP)
                    
            amino_acids_ap_padded = padding(amino_acids_ap, len(encoded_sequences[index]), masking_value)
            dipeptides_acids_ap_padded = padding(dipeptides_ap, len(encoded_sequences[index]), masking_value)
            tripeptides_ap_padded = padding(tripeptides_ap, len(encoded_sequences[index]), masking_value)  

            new_props.append(amino_acids_ap_padded)
            new_props.append(dipeptides_acids_ap_padded)
            new_props.append(tripeptides_ap_padded) 

            #new_props = read_mordred(sequences[index], new_props, len(encoded_sequences[index]), masking_value)

        other_props = np.transpose(encoded_sequences[index])  

        for prop_index in range(len(properties_to_include)):
            if prop_index < len(other_props) and properties_to_include[prop_index] == 1:
                array = other_props[prop_index]
                for i in range(len(array)):
                    if i >= len(sequences[index]):
                        array[i] = masking_value
                new_props.append(array)
                 
        new_props = np.transpose(new_props) 

        if labels[index] == '1':
            SA.append(new_props) 
        elif labels[index] == '0':
            NSA.append(new_props) 
    if len(SA) > 0:
        SA = np.array(SA)
    if len(NSA) > 0:
        NSA = np.array(NSA)
    return SA, NSA

def load_data_SA(SA_data, names=['AP'], offset = 1, properties_to_include = [], masking_value=2):
    sequences, labels = read_from_npy_SA(SA_data)
            
    # Encode sequences
    encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-offset, offset)))
    encoded_sequences = encoder.encode(sequences)
     
    # Split peptides in two bins.
    # SA - has self-assembly, NSA - does not have self-assembly.
    # Convert each peptide sequence to three lists of AP scores: amino acid, dipeptide and tripeptide scores.
    SA = []
    NSA = []
    for index in range(len(sequences)):
        new_props = []
        for name in names:
            amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP(name, offset)  
            amino_acids_ap = split_amino_acids(sequences[index], amino_acids_AP)
            dipeptides_ap = split_dipeptides(sequences[index], dipeptides_AP)
            tripeptides_ap = split_tripeptides(sequences[index], tripeptides_AP)
                    
            amino_acids_ap_padded = padding(amino_acids_ap, len(encoded_sequences[index]), masking_value)
            dipeptides_acids_ap_padded = padding(dipeptides_ap, len(encoded_sequences[index]), masking_value)
            tripeptides_ap_padded = padding(tripeptides_ap, len(encoded_sequences[index]), masking_value)  
        
            new_props.append(amino_acids_ap_padded)
            new_props.append(dipeptides_acids_ap_padded)
            new_props.append(tripeptides_ap_padded)

            #new_props = read_mordred(sequences[index], new_props, len(encoded_sequences[index]), masking_value)
        
        other_props = np.transpose(encoded_sequences[index])  

        for prop_index in range(len(properties_to_include)):
            if prop_index < len(other_props) and properties_to_include[prop_index] == 1:
                array = other_props[prop_index]
                for i in range(len(array)):
                    if i >= len(sequences[index]):
                        array[i] = masking_value
                new_props.append(array) 
        
        if labels[index] == '1':
            SA.append(new_props) 
        elif labels[index] == '0':
            NSA.append(new_props) 
            
    return SA, NSA

def load_data_SA_AP(SA_data, names=['AP'], offset = 1, properties_to_include = [], masking_value=2):
    sequences, labels = read_from_npy_SA(SA_data)
     
    # Split peptides in two bins.
    # SA - has self-assembly, NSA - does not have self-assembly.
    # Convert each peptide sequence to three lists of AP scores: amino acid, dipeptide and tripeptide scores.
    SA = []
    NSA = []
    for index in range(len(sequences)):
        new_props = []
        for name in names:
            amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP(name, offset)  
            amino_acids_ap = split_amino_acids(sequences[index], amino_acids_AP)
            dipeptides_ap = split_dipeptides(sequences[index], dipeptides_AP)
            tripeptides_ap = split_tripeptides(sequences[index], tripeptides_AP)
                    
            amino_acids_ap_padded = padding(amino_acids_ap, MAXLEN, masking_value)
            dipeptides_acids_ap_padded = padding(dipeptides_ap, MAXLEN, masking_value)
            tripeptides_ap_padded = padding(tripeptides_ap, MAXLEN, masking_value)  
        
            new_props.append(amino_acids_ap_padded)
            new_props.append(dipeptides_acids_ap_padded)
            new_props.append(tripeptides_ap_padded) 

            #new_props = read_mordred(sequences[index], new_props, MAXLEN, masking_value)
        
        if labels[index] == '1':
            SA.append(new_props) 
        elif labels[index] == '0':
            NSA.append(new_props) 
            
    return SA, NSA

def merge_data(SA, NSA):
    # Merge the bins and add labels
    merged_data = []
    for i in SA:
        merged_data.append(i)
    for i in NSA:
        merged_data.append(i)

    merged_labels = np.ones(len(SA) + len(NSA))
    merged_labels[len(SA):] *= 0
    return merged_data, merged_labels
   
def merge_data_AP(SA, NSA):
    # Merge the bins and add labels
    merged_data = []
    for i in SA:
        merged_data.append(i)
    for i in NSA:
        merged_data.append(i)

    merged_labels = np.ones(len(SA) + len(NSA))
    merged_labels[len(SA):] *= 0
    return merged_data, merged_labels

def merge_data_seq(SA, NSA):
    # Merge the bins and add labels
    merged_data = []
    for i in SA:
        merged_data.append(i)
    for i in NSA:
        merged_data.append(i)
    if len(merged_data) > 0:
        merged_data = np.array(merged_data)
        #merged_data = np.reshape(merged_data, (len(merged_data), np.shape(merged_data[0])[0], np.shape(merged_data[0])[1]))
    merged_labels = np.ones(len(SA) + len(NSA))
    merged_labels[len(SA):] *= 0

    return merged_data, merged_labels