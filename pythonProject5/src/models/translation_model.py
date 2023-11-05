import random
# create seq2seq model with attention, encoder and decoder
import torch.nn as nn
import torch
import spacy
import pickle
from transformers import pipeline


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, source_sequence, source_mask):
        # source_sequence shape: (source_sequence_length, batch_size)
        # source_mask shape: (source_sequence_length, batch_size)

        embedded = self.dropout(self.embedding(source_sequence))
        # embedded shape: (source_sequence_length, batch_size, embedding_size)

        packed = nn.utils.rnn.pack_padded_sequence(embedded, source_mask.cpu().sum(0).long(), enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed)
        # outputs shape: (source_sequence_length, batch_size, hidden_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell shape: (num_layers, batch_size, hidden_size)

        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, target_sequence, hidden, cell, encoder_outputs):
        # target_sequence shape: (batch_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell shape: (num_layers, batch_size, hidden_size)
        # encoder_outputs shape: (source_sequence_length, batch_size, hidden_size)

        target_sequence = target_sequence.unsqueeze(0)
        # target_sequence shape: (1, batch_size)

        hidden = hidden.view(self.num_layers, target_sequence.size(1), self.hidden_size)
        cell = cell.view(self.num_layers, target_sequence.size(1), self.hidden_size)

        embedded = self.dropout(self.embedding(target_sequence))
        # embedded shape: (target_sequence_length, batch_size, embedding_size)

        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output shape: (1, batch_size, hidden_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell shape: (num_layers, batch_size, hidden_size)

        prediction = self.fc(outputs.squeeze(0))
        # prediction shape: (batch_size, input_size)

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source_sequence, source_mask, target_sequence):
        # source_sequence shape: (source_sequence_length, batch_size)
        # source_mask shape: (source_sequence_length, batch_size)
        # target_sequence shape: (target_sequence_length, batch_size)

        batch_size = source_sequence.shape[1]
        target_sequence_length = target_sequence.shape[0]
        target_vocab_size = self.decoder.input_size

        outputs = torch.zeros(target_sequence_length, batch_size, target_vocab_size).to(self.device)
        # outputs shape: (target_sequence_length, batch_size, target_vocab_size)

        encoder_outputs, hidden, cell = self.encoder(source_sequence, source_mask)
        # encoder_outputs shape: (source_sequence_length, batch_size, hidden_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell shape: (num_layers, batch_size, hidden_size)

        # First input to the decoder is the <SOS> tokens
        input = target_sequence[0, :]

        for t in range(1, target_sequence_length):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            # output shape: (batch_size, target_vocab_size)
            outputs[t] = output
            # outputs shape: (target_sequence_length, batch_size, target_vocab_size)

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability 0.5 we take the actual next word in the sequence
            # otherwise we take the word that the Decoder predicted it to be.
            input = target_sequence[t] if random.random() < 0.5 else best_guess

        return outputs

    def predict(self, source_sequence, source_mask, target_vocab_to_int, int_to_target_vocab):
        # source_sequence shape: (source_sequence_length, batch_size)
        # source_mask shape: (source_sequence_length, batch_size)

        batch_size = source_sequence.shape[1]
        target_vocab_size = self.decoder.input_size
        max_target_length = 100

        outputs = torch.zeros(max_target_length, batch_size, target_vocab_size).to(self.device)
        # outputs shape: (target_sequence_length, batch_size, target_vocab_size)

        encoder_outputs, hidden, cell = self.encoder(source_sequence, source_mask)
        # encoder_outputs shape: (source_sequence_length, batch_size, hidden_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell shape: (num_layers, batch_size, hidden_size)

        # First input to the decoder is the <SOS> tokens
        input = torch.tensor([target_vocab_to_int['<SOS>']] * batch_size).to(self.device)

        for t in range(1, max_target_length):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            # output shape: (batch_size, target_vocab_size)
            outputs[t] = output
            # outputs shape: (target_sequence_length, batch_size, target_vocab_size)

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability 0.5 we take the actual next word in the sequence
            # otherwise we take the word that the Decoder predicted it to be.
            input = best_guess

        # Remove <SOS> token
        outputs = outputs[1:]
        # outputs shape: (target_sequence_length, batch_size, target_vocab_size)

        # Get the best word indices (indexes in the vocabulary) per time step
        best_guess = outputs.argmax(2)
        # best_guess shape: (target_sequence_length, batch_size)

        # Convert the indices into actual words
        decoded_words = []
        for i in range(best_guess.shape[1]):
            predicted_sentence = [int_to_target_vocab[int_.item()] for int_ in best_guess[:, i]]
            decoded_words.append(' '.join(predicted_sentence))

        return decoded_words


class TranslationModel:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder_embedding_size = 256
        self.decoder_embedding_size = 256
        self.hidden_size = 1024
        self.num_layers = 4
        self.encoder_dropout = 0.2
        self.decoder_dropout = 0.2

        # load all 4 vocabularies

        with open('../../models/source_vocab_to_int.pickle', 'rb') as handle:
            self.source_vocab_to_int = pickle.load(handle)

        with open('../../models/target_vocab_to_int.pickle', 'rb') as handle:
            self.target_vocab_to_int = pickle.load(handle)

        with open('../../models/int_to_source_vocab.pickle', 'rb') as handle:
            self.int_to_source_vocab = pickle.load(handle)

        with open('../../models/int_to_target_vocab.pickle', 'rb') as handle:
            self.int_to_target_vocab = pickle.load(handle)

        self.input_size_encoder = len(self.source_vocab_to_int)
        self.input_size_decoder = len(self.target_vocab_to_int)

        # Initialize network
        self.encoder_net = Encoder(self.input_size_encoder,
                                   self.hidden_size,
                                   self.encoder_embedding_size,
                                   self.num_layers,
                                   self.encoder_dropout)

        self.decoder_net = Decoder(self.input_size_decoder,
                                   self.hidden_size,
                                   self.decoder_embedding_size,
                                   self.num_layers,
                                   self.decoder_dropout)

        self.model = Seq2Seq(self.encoder_net,
                             self.decoder_net,
                             self.device).to(self.device)

        self.model.load_state_dict(torch.load('../../models/seq2seq_model.pt'))

        self.nlp = spacy.load('en_core_web_sm')

    def tokenize(self, text):
        return [token.text.lower()
                for token in self.nlp.tokenizer(text)
                if not token.is_space]

    def prepare_sent(self, sentence: str):
        sentence = self.tokenize(sentence)

        sentence = [self.source_vocab_to_int.get(word, self.source_vocab_to_int['<UNK>'])
                    for word in sentence]

        sentence = torch.tensor(sentence).unsqueeze(1).to(self.device)
        return sentence

    def translate(self, sentence):
        sent = self.prepare_sent(sentence)
        self.model.eval()
        with torch.no_grad():
            translation = self.model.predict(sent,
                                             torch.ones(sent.shape).to(self.device),
                                             self.target_vocab_to_int,
                                             self.int_to_target_vocab)

        words = list(set(translation[0].split()))

        return words




