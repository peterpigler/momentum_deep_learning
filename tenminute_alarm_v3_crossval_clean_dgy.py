from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import math
#visualization utils
from keras.utils import plot_model
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from keras.utils import np_utils
start_time = time.time()
emb_dim = 40
latent_dim = 256
val1s = []
val2s = []
val1_tsts = []
val2_tsts = []


batch_size = 256  # Batch size for training. 64
epochs = 4000  # Number of epochs to train for.
#num_samples = 10000  # Number of samples to train on.
length_min_seq = 5 # minimal sequence length - while choosing take care: the target length will be split from this length
where_to_split = 3 # split, counted from the end
##################################################################################################################
# Vectorize the data.
seq_type = 0 # NEW datasets!!!
if seq_type == 0:
            df_sequences = pd.read_excel('Sequences_alarm_warning_171208.xlsx', header=None, dtype=float)
            EOS = df_sequences.values.min() # end of sequences erteke
            PAD = EOS - 2  # padding erteke, PAD legyen a legkisebb, átkódolva 0 lesz
            StOS = EOS - 1 # Start of Sequence jel
            df_sequences.replace(EOS,PAD, inplace=True) # az eredtileg paddingelt szekvenciak pad erteket lecsokkentem, helyet csinalva a EOS-nak
            sequences = df_sequences.as_matrix() # numpy array a DF-bol
            sequences = pd.DataFrame(sequences[:,::2]) # vagjuk ki a temporalokat, rakjuk vissza df-be
            sequences[str(sequences.shape[1])] = PAD # a leghosszabb szekvenciak vegere nem tudunk EOS-t rakni, ezert kell meg egy oszlop, alapertek = -6.0
            sequences = sequences.as_matrix() # vissza numpy arraybe ... maceras!
        
            input_seqs = []
            target_seqs = []
            for row in range(sequences.shape[0]):
                length_sequence = list(sequences[row]).index(PAD) # the end of the sequence
                if length_sequence > length_min_seq:
                    input_seq = sequences[row,:math.ceil(length_sequence/2)]
                    target_seq = sequences[row,math.ceil(length_sequence/2):]
                    target_seq_end = list(target_seq).index(PAD)
                    target_seq[target_seq_end] = EOS
                    target_seq = np.insert(target_seq, 0, StOS)
                    input_seqs.append(input_seq)
                    target_seqs.append(target_seq)
            input_seqs = np.array(input_seqs)
            target_seqs = np.array(target_seqs)
            
            df_faultclass = pd.read_excel('Fault_0822.xlsx', header=None, dtype=int)
            faultclass = df_faultclass.as_matrix()
            shape_faultclass = faultclass.shape
            YEncoder = preprocessing.LabelEncoder()
            YEncoder.fit(faultclass.flatten()) 
            Y_encoded = YEncoder.fit_transform(faultclass)
            y_label=Y_encoded
            faultclass=Y_encoded.reshape((shape_faultclass)) 
            YEncoder_OH = preprocessing.LabelBinarizer()
            YEncoder_OH.fit(faultclass)
            faultclass = YEncoder_OH.transform(faultclass)
            #faultclass = np_utils.to_categorical(faultclass)
    #a = 5

else:
    
    df_input_sequences = pd.read_excel('Input_sequences_alarmwarning_SIMAconf0.8_171213.xlsx', header=None, dtype=float)
    EOS = df_input_sequences.values.min() # end of sequences erteke
    PAD = EOS - 2  # padding erteke, PAD legyen a legkisebb, átkódolva 0 lesz
    StOS = EOS - 1 # Start of Sequence jel
    df_input_sequences.replace(EOS,PAD, inplace=True) # az eredtileg paddingelt szekvenciak pad erteket lecsokkentem, helyet csinalva a EOS-nak
    df_input_sequences[str(df_input_sequences.shape[1])] = PAD # a leghosszabb szekvenciak vegere nem tudunk EOS-t rakni, ezert kell meg egy oszlop, alapertek = -6.0
    input_sequences = df_input_sequences.as_matrix() # vissza numpy arraybe ... maceras!


    df_target_sequences = pd.read_excel('Target_sequences_alarmwarning_SIMAconf0.8_171213.xlsx', header=None, dtype=float)
    df_target_sequences.replace(EOS,PAD, inplace=True) # az eredtileg paddingelt szekvenciak pad erteket lecsokkentem, helyet csinalva a EOS-nak
    df_target_sequences[str(df_target_sequences.shape[1])] = PAD # a leghosszabb szekvenciak vegere nem tudunk EOS-t rakni, ezert kell meg egy oszlop, alapertek = -6.0
    target_sequences = df_target_sequences.as_matrix() # vissza numpy arraybe ... maceras!

    input_seqs = []
    target_seqs = []
    for row in range(input_sequences.shape[0]):
        length_sequence = list(input_sequences[row]).index(PAD) # the end of the sequence
        input_seq = input_sequences[row,:length_sequence]
        target_seq = target_sequences[row]
        target_seq_end = list(target_seq).index(PAD)
        target_seq[target_seq_end] = EOS
        target_seq = np.insert(target_seq, 0, StOS)
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
    input_seqs = np.array(input_seqs)
    target_seqs = np.array(target_seqs)


##################################################################################################################
df_sequences_nf = pd.read_excel('Sequences_alarm_warning_171208.xlsx', header=None, dtype=float)
input_characters = np.array(df_sequences_nf)
input_characters = np.append(input_characters, [EOS, PAD, StOS])
input_characters = np.unique(input_characters)
df_sequences_nf.replace(EOS,PAD, inplace=True) # az eredtileg paddingelt szekvenciak pad erteket lecsokkentem, helyet csinalva a EOS-nak
sequences_nf = df_sequences_nf.as_matrix() # numpy array a DF-bol
sequences_nf = pd.DataFrame(sequences_nf[:,::2]) # vagjuk ki a temporalokat, rakjuk vissza df-be
sequences_nf[str(sequences_nf.shape[1])] = PAD # a leghosszabb szekvenciak vegere nem tudunk EOS-t rakni, ezert kell meg egy oszlop, alapertek = -6.0
sequences_nf = sequences_nf.as_matrix() # vissza numpy arraybe ... maceras!

input_seqs_nf = []
target_seqs_nf = []
target_classes_nf = []
for row in range(sequences_nf.shape[0]):
    current_faultclass = y_label[row]
    length_sequence = list(sequences_nf[row]).index(PAD) # the end of the sequence
    if length_sequence > length_min_seq:
        input_seq = sequences_nf[row,:length_sequence-where_to_split]
        target_seq = sequences_nf[row,length_sequence-where_to_split:]
        target_seq_end = list(target_seq).index(PAD)
        target_seq[target_seq_end] = EOS
        target_seq = np.insert(target_seq, 0, StOS)
        input_seqs_nf.append(input_seq)
        target_seqs_nf.append(target_seq)
        target_classes_nf.append(current_faultclass)
input_seqs_nf = np.array(input_seqs_nf)
target_seqs_nf = np.array(target_seqs_nf)
faultclass = np.array(target_classes_nf)

##################################################################################################################
target_characters = input_characters
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(input_seq) for input_seq in input_seqs])
max_decoder_seq_length = max([len(target_seq) for target_seq in target_seqs])
max_encoder_seq_length_nf = max([len(input_seq) for input_seq in input_seqs_nf])
max_decoder_seq_length_nf = max([len(target_seq) for target_seq in target_seqs_nf])

if max_encoder_seq_length_nf > max_encoder_seq_length:
    max_encoder_seq_length = max_encoder_seq_length_nf
if max_decoder_seq_length_nf > max_decoder_seq_length:
    max_decoder_seq_length = max_decoder_seq_length_nf


print('Number of samples:', len(input_seqs))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = input_token_index

encoder_input_data = np.zeros(
    (len(input_seqs), max_encoder_seq_length),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_seqs), max_decoder_seq_length),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_seqs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

REV_SEQ = True
for i, (input_seq, target_seq) in enumerate(zip(input_seqs, target_seqs)):
    for t, char in enumerate(input_seq):
        if REV_SEQ == True:
            encoder_input_data[i, -(t+1)] = input_token_index[char]
        else:
            encoder_input_data[i, t] = input_token_index[char]
    for t, char in enumerate(target_seq):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = target_token_index[char]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


##################################################################################################################
# eddig kell eljutnunk ;)
# Define an input sequence and process it.
SSS = StratifiedShuffleSplit(n_splits=7, test_size=0.2, random_state=42)

results_val1 = []
results_val2 = []
models = []
overall_results = []
overall_results_crossval = []
for train_index, test_index in SSS.split(encoder_input_data, faultclass):
    encoder_inputs = Input(shape=(None,), name='enc_input')
    x= Embedding(num_encoder_tokens, emb_dim, name='enc_emb')(encoder_inputs)
    x, state_h, state_c = LSTM(latent_dim,
                               return_state=True, name='enc')(x)
    encoder_states = [state_h, state_c]
    
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,), name='dec_input')
    emb = Embedding(num_decoder_tokens, emb_dim, name='dec_emb')
    x = emb(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='dec')
    decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
    decoder_dense =  Dense(num_decoder_tokens, activation='softmax', name='dec_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    
    model.summary()
    #plot_model(model, to_file='model.png', show_shapes =True)
    # Compile & run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # Note that `decoder_target_data` needs to be one-hot encoded,
    # rather than sequences of integers like `decoder_input_data`!
    
    
    model.fit([encoder_input_data[train_index], decoder_input_data[train_index]], decoder_target_data[train_index],
              batch_size=batch_size,
              epochs=epochs,
              validation_data=([encoder_input_data[test_index], 
                                decoder_input_data[test_index]
                                ], 
                                decoder_target_data[test_index]),
              verbose=1)
    
    
    #model.load_weights('MENTETT MODELLEK/tenminute_alarm_v3_sima/weights_model_12_15_1422.h5')
    # Save model
    model.save('seq2seq_alarm_4000epoch_FINAL.h5')
    
    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states
    
    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states, name='enc_b')
    #The encoder model is defined as taking the input layer from the encoder 
    #in the trained model (encoder_inputs) and outputting the hidden and 
    #cell state tensors (encoder_states).
    #plot_model(encoder_model, to_file='model_enc.png', show_shapes =True)
    encoder_model.summary()
    #encoder_model.load_weights('MENTETT MODELLEK/tenminute_alarm_v3_sima/weights_encoder_12_15_1422.h5')
    decoder_state_input_h = Input(shape=(latent_dim,), name='input_h')
    decoder_state_input_c = Input(shape=(latent_dim,), name='input_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    x = emb(decoder_inputs)
    decoder_outputs, state_h, state_c = decoder_lstm(
        x, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states, name='dec_b')
    #plot_model(decoder_model, to_file='model_dec.png', show_shapes =True)
    decoder_model.summary()
    #decoder_model.load_weights('MENTETT MODELLEK/tenminute_alarm_v3_sima/weights_decoder_12_15_1422.h5')
    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())
    
    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)
    
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = target_token_index[StOS]
    
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_seq = []
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)
    
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_seq.append(sampled_char)
            
            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == EOS or
               len(decoded_seq) > max_decoder_seq_length):
                stop_condition = True
    
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
    
            # Update states
            states_value = [h, c]
        decoded_seq = np.array(decoded_seq)
        return decoded_seq
    
    analysed_seqs = np.random.randint(len(input_seqs), size=20)
    for num in range(len(analysed_seqs)):
        # Take one sequence (part of the training test)
        # for trying out decoding.
        seq_index = analysed_seqs[num]
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_seq = decode_sequence(input_seq)
        print('-')
        print('Input sequence:', input_seqs[seq_index])
        print('Decoded sequence:', decoded_seq)
        if PAD in target_seqs[seq_index]:
            length_sequence = list(target_seqs[seq_index]).index(PAD)
            print('True sequence:', target_seqs[seq_index][1:length_sequence])
        else:
            length_sequence = list(target_seqs[seq_index]).index(EOS)
            print('True sequence:', target_seqs[seq_index][1:])    
       
    
    val1 = []
    val2 = []
    val1s = []
    val2s = []
    for seq_index in range(len(encoder_input_data)):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_seq = decode_sequence(input_seq)
        decoded_seq = decoded_seq[:-1]
        orig_input_seq = input_seqs[seq_index]
        if PAD in target_seqs[seq_index]:
            length_sequence = list(target_seqs[seq_index]).index(PAD)
            true_seq = target_seqs[seq_index][1:length_sequence]
        else:
            length_sequence = list(target_seqs[seq_index]).index(EOS)
            true_seq = target_seqs[seq_index][1:] 
        true_seq = true_seq[:-1]
        tmp = []
        for i in range(len(decoded_seq)):
            if decoded_seq[i]  in true_seq:
                val1.append(seq_index)
                tmp.append(i)
        val2.append(len(tmp)/len(true_seq))
        val2s.append(len(tmp)/len(true_seq))
    val1 = np.array(val1)
    val1s.append(np.unique(val1))
    val1 = len(np.unique(val1))/len(encoder_input_data)
    val2 = np.mean(val2)
    overall_results.append([val1, val2])
    overall_results_crossval = np.append(overall_results_crossval, [val1s, val2s])
#        
    
    
    
    
    
    

    result_1val = []
    result_2val = []
    for fault in range(max(faultclass)+1):
        fault_indices = np.where(faultclass==fault)[0]
        print(len(fault_indices))
        encoder_input_data_split = encoder_input_data[fault_indices]
        target_seqs_split = target_seqs[fault_indices]
        val1_tst = []
        val2_tst = []    
        for seq_index in range(len(encoder_input_data_split)):
            input_seq = encoder_input_data_split[seq_index: seq_index + 1]
            decoded_seq = decode_sequence(input_seq)
            decoded_seq = decoded_seq[:-1]
            if PAD in target_seqs_split[seq_index]:
                length_sequence = list(target_seqs_split[seq_index]).index(PAD)
                true_seq = target_seqs_split[seq_index][1:length_sequence]
            else:
                length_sequence = list(target_seqs_split[seq_index]).index(EOS)
                true_seq = target_seqs_split[seq_index][1:] 
            true_seq = true_seq[:-1]
            tmp = []
            for i in range(len(decoded_seq)):
                if decoded_seq[i]  in true_seq:
                    val1_tst.append(seq_index)
                    tmp.append(i)
            val2_tst.append(len(tmp)/len(true_seq))
        val1_tst = np.array(val1_tst)
        if len(encoder_input_data_split) != 0:
            val1_tst = len(np.unique(val1_tst))/len(encoder_input_data_split)
        else:
            val1_tst = 0        
        val2_tst= np.mean(val2_tst)
        result_1val.append(val1_tst)
        result_2val.append(val2_tst)
    
    
    results_val1.append(result_1val)
    results_val2.append(result_2val)
    
    
    
results_val1 = np.array(results_val1)
results_val2 = np.array(results_val2)  


fig = plt.figure() 
ax = fig.add_subplot(211)
bp = ax.boxplot(results_val1*100)
plt.ylabel('Corr. Pred. Rate (%)')
plt.title('Pred. Seqs with good event in them',Fontsize=8)
#ax.set_xticklabels(VnEVENT)
ax.set_xticklabels([])
plt.ylim((80,100))

ax = fig.add_subplot(212)
plt.title('Percent of correctly predicted events in seq.',Fontsize=8)
bp = ax.boxplot(results_val2*100)
plt.ylabel('Corr. Pred. Rate (%)')
#ax.set_xticklabels(VnEVENT)
plt.xlabel('# of Fault')
#plt.ylim((50,87))
    
plt.savefig('VAL1_VAL2_byrfaults.png', dpi=300)
plt.show()


overall_results = np.array(overall_results)
fig = plt.figure() 
ax = fig.add_subplot(111)
bp = ax.boxplot(overall_results*100)
plt.ylabel('Corr. Pred. Rate (%)')
plt.title('Pred. Seqs with good event in them',Fontsize=8)
#ax.set_xticklabels(VnEVENT)
ax.set_xticklabels(['val1', 'val2'])
#plt.ylim((80,100))

plt.savefig('VAL1_VAL2_overall.png', dpi=300)
plt.show()


adR1 = pd.DataFrame(results_val1)
adR2 = pd.DataFrame(results_val2)

writer = pd.ExcelWriter('val1_val2.xlsx')
adR1.to_excel(writer,'val1')
adR2.to_excel(writer,'val2')
writer.save()




results_all1 = np.insert(results_val1, 0, overall_results[:, 0],  axis=1)
results_all2 = np.insert(results_val2, 0, overall_results[:, 1],  axis=1)

adR1 = pd.DataFrame(results_all1)
adR2 = pd.DataFrame(results_all2)
writer = pd.ExcelWriter('val1_val2.xlsx')
adR1.to_excel(writer,'val1')
adR2.to_excel(writer,'val2')
writer.save()

results_all1 = np.delete(results_all1, 8, 1)
results_all2 = np.delete(results_all2, 8, 1)
xlabels = ['Overall', 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]

fig = plt.figure() 
ax = fig.add_subplot(211)
bp = ax.boxplot(results_all1*100)
plt.ylabel('$Val_1$ (%)')
#plt.title('Pred. Seqs with good event in them',Fontsize=8)
#ax.set_xticklabels(VnEVENT)
ax.set_xticklabels([])
#plt.ylim((80,100))

ax = fig.add_subplot(212)
#plt.title('Percent of correctly predicted events in seq.',Fontsize=8)
bp = ax.boxplot(results_all2*100)
plt.ylabel('$Val_2$ (%)')

ax.set_xticklabels(xlabels)
plt.xlabel('# of Fault')
#plt.ylim((50,87))
    
plt.savefig('VAL1_VAL2_byfaults.png', dpi=300)
plt.show()



elapsed_time = time.time() - start_time



























