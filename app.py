import time
import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved models
encoder_model = load_model("encoder_model.keras")
decoder_model = load_model("decoder_model.keras")

# Load saved dictionaries
with open('hindi_word_index.pickle', 'rb') as f:
    hindi_word_index = pickle.load(f)

with open('hindi_index_word.pickle', 'rb') as f:
    hindi_index_word = pickle.load(f)

with open('english_word_index.pickle', 'rb') as f:
    english_word_index = pickle.load(f)

with open('english_index_word.pickle', 'rb') as f:
    english_index_word = pickle.load(f)

# Load the hyperparameters
max_eng_sen_len = 20
max_hindi_sen_len = 20

###############################################################################################################################
############################### Function to translate the input sequence into the hindi text ################################## 
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = hindi_word_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = hindi_index_word[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 50):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

###############################################################################################################################
###############################################################################################################################

# Set up the page configuration
st.set_page_config(
    page_title="English to Hindi Translator", 
    page_icon="📰", 
    layout="wide"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
        .main-title {
            font-size: 40px;
            text-align: center;
            color: #3E4C59;
        }
        .section-title {
            font-size: 24px;
            color: #3E4C59;
        }
        textarea {
            font-size: 16px;
            color: black; /* Make the text darker */
            background-color: #f7f9fc; /* Optional lighter background for text box */
            font-weight: bold; /* Make text bold */
        }
        button {
            font-size: 14px;
            background-color: #007BFF;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add icons side by side
st.markdown(
    """
    <style>
        .top-right-icons {
            position: absolute;
            top: 10px; /* Adjust for vertical placement */
            right: 10px; /* Adjust for horizontal placement */
            display: flex;
            align-items: center;
            gap: 10px; /* Space between icons */
        }
        .top-right-icons img {
            height: 20px; /* Icon size */
        }
    </style>
    <div class="top-right-icons">
        <a href="https://github.com/abdul-random" target="_blank">
            <img src="https://badgen.net/badge/icon/GitHub?icon=github&label" alt="GitHub Repo">
        </a>
        <a href="https://www.linkedin.com/in/abdulshaik12/" target="_blank">
            <img src="https://badgen.net/badge/icon/LinkedIn?icon=linkedin&label" alt="LinkedIn Profile">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


# Title of the app
st.markdown('<h1 class="main-title">📰 English to Hindi Translator using Seq2Seq Model (Encoder+Decoder)</h1>', unsafe_allow_html=True)

# Define headers and emojis
english_header = "English Input"
hindi_header = "Hindi Output"

# Initialize two columns for the input and output boxes
col1, col2 = st.columns(2)

# Create placeholders for dynamic updates
with col1:
    st.header(english_header)
    english_text = st.text_area("Enter English text here:", value = 'There are fifteen thousand verses in the agnipuran', height=200)
    translate_button = st.button("Translate")

with col2:
    st.header(hindi_header)
    hindi_text_placeholder = st.empty()  # Placeholder to update Hindi box dynamically
    hindi_text = hindi_text_placeholder.text_area(
        "Translated Hindi text will appear here:", height=200, disabled=True
    )

# Process the translation
if translate_button:
    if english_text.strip():  # Check if the input is not empty
        # Show a "loading indicator" inside the Hindi output box

        hindi_text_placeholder.text_area(
            "Translated Hindi text will appear here:", value="⏳ Translating...", height=200, disabled=True
        )

        try:
            trim_sentence = english_text.lower().split()[:19]  # Limit to first 19 words
            seq = np.zeros((1, max_eng_sen_len))
            # Convert words to indices
            for t, word in enumerate(trim_sentence):
                seq[0, t] = english_word_index.get(word, 0)  # Assign 0 if word not found
            # Generate the translated sentence
            translated_sentence = decode_sequence(seq)
        except:
            pass

        # Update the Hindi text area dynamically after "translation"
        hindi_translated = translated_sentence[:-4]  # Replace this with actual translation logic
        hindi_text_placeholder.text_area(
            "Translated Hindi text will appear here:", value=hindi_translated, height=200, disabled=False
        )
        
    else:
        st.warning("Please enter text in the English Input box.")


st.markdown(
    """<p class="warning" style="color: #cc0000; font-size: 20px; text-align: center; font-weight: bold;">
    ⚠️ Note: This is not a perfect model as the number of data points it has been trained on is just 10,000. 
    The weights and biases considered are very less, as LSTM units and embedding sizes are just 64. 
    Epochs are also just 500. Therefore, outputs are not accurate as test accuracy is very low. 
    I will try to improve the model later.<br/>
    Data source: <a href="https://www.kaggle.com/code/aiswaryaramachandran/english-to-hindi-neural-machine-translation/input" target="_blank" style="color: #cc0000;">Link</a><br/>
    Reference: <a href="https://arxiv.org/pdf/1409.3215" target="_blank" style="color: #cc0000;">Link</a>
    <a href="https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html"target="_blank" style="color: #cc0000;">Link</a>
    <a href="https://medium.com/analytics-vidhya/machine-translation-encoder-decoder-model-7e4867377161" target="_blank" style="color: #cc0000;">Link</a>
    </p>""", 
    unsafe_allow_html=True
)
st.markdown("<h3 class='section-title'>Sample Sentences for Testing:</h3>", unsafe_allow_html=True)
st.markdown("1. Is transmitted generation from generation")
st.markdown("2. There are fifteen thousand verses in the agnipuran")
st.markdown("3. But kids could be entrepreneurs as well")

st.markdown("<h3 class='section-title'>Seq2Seq Model (Encoder + Decoder)</h3>", unsafe_allow_html=True)
st.image("Encoder_and_Decoder.jpg")
st.markdown("The Seq2Seq (Sequence-to-Sequence) model is designed to transform one sequence into another sequence, accommodating varying lengths for both input and output. It consists of two primary components: the encoder and the decoder. The encoder reads the input sequence and compresses it into a fixed-length context vector or a series of hidden states. This encoding captures the essential information necessary to understand the input sequence. The decoder then takes this encoded information — the context vector — and generates the output sequence. By leveraging the context provided by the encoder, the decoder constructs the output step by step, ensuring that the generated sequence is coherent and aligned with the input sequence's meaning. Seq2Seq models are particularly useful in tasks such as machine translation, where they convert sentences from one language to another, and time-series forecasting, where they predict future events based on past data.")

st.markdown("### Model Development Code:")
st.code("""
# Encoder and Decoder Architecture
embedding_dim = 64
lstm_dim = 64

# Encoder
encoder_inputs = Input(shape=(None,), name='encoder_inputs')
encoder_emb = Embedding(input_dim=english_vocab_len, output_dim=embedding_dim, mask_zero=True, name='encoder_embedding')(encoder_inputs)
encoder_lstm = LSTM(lstm_dim, name='encoder_lstm', return_state=True)
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_emb)
encoder_state = [encoder_state_h, encoder_state_c]

# Decoder
decoder_inputs = Input(shape=(None,), name='decoder_inputs')
decoder_emb_layer = Embedding(input_dim=hindi_vocab_len, output_dim=embedding_dim, mask_zero=True, name='decoder_embedding')
decoder_emb = decoder_emb_layer(decoder_inputs)
decoder_lstm = LSTM(lstm_dim, name='decoder_lstm', return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=encoder_state)

decoder_dense = Dense(hindi_vocab_len, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Complie the model with optimizer rmsprop and loss categorical_crossentropy as we have multi-class classification problems 
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')      

# Training the model 
batch_size = 128
epochs = 500

history = model.fit([encoder_input_data,decoder_input_data],
                    decoder_target_data, 
                    epochs=epochs,
                    # batch_size = batch_size,
                    )   
""", language="python")

st.markdown("### Model Inference Code:")
st.code("""
# Model inference code
# As Training and Predict is slightly Diffrent in terms of Architecture
# Encode the input sequence to get the "thought vectors"
encoder_model = Model(encoder_inputs, encoder_state)

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(lstm_dim,))
decoder_state_input_c = Input(shape=(lstm_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2= decoder_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)
""", language="python")

