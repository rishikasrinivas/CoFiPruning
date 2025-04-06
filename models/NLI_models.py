  
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np

import collections
from typing import Union
from transformers import AutoTokenizer, AutoModel
from llm2vec.models import LlamaBiModel

class SentimentClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder
        self.encoder_dim = encoder.output_dim
        self.mlp_input_dim = self.encoder_dim
        self.dropout = nn.Dropout(0.5)

        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 1),
        )

        self.output_dim = 1

    def forward(self, text, length):
        enc = self.encoder(text, length)
        enc = self.dropout(enc)

        logits = self.mlp(enc)
        logits = logits.squeeze(1)
        return logits

    def get_final_reprs(self, text, length):
        """
        Get features right up to final decision
        """
        enc = self.encoder(text, length)
        rep = self.mlp[:-1](enc)
        return rep


class EntailmentClassifier(nn.Module):
    """
    An NLI entailment classifier where the hidden rep features are much
    "closer" to the actual feature decision
    """
    #look into how vocab size affects the model. rnn weights are the same beforea fter pruing but vocab size differs. before its 33587 after is 5784 something

    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder
        self.encoder_dim = encoder.output_dim
        self.mlp_input_dim = self.encoder_dim
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(self.mlp_input_dim)

        self.mlp = nn.Linear(self.mlp_input_dim, 3)

        self.output_dim = 3

    def forward(self, s1, s1len, s2, s2len):
        s1enc = self.encoder(s1, s1len)
        s2enc = self.encoder(s2, s2len)

        mlp_input = s1enc * s2enc

        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)

        preds = self.mlp(mlp_input)

        return preds

    def get_final_reprs(self, s1, s1len, s2, s2len):
        """
        Get features right up to final decision
        """
        s1enc = self.encoder(s1, s1len)
        s2enc = self.encoder(s2, s2len)
        mlp_input = s1enc * s2enc

        return mlp_input
#https://github.com/lecode-official/pytorch-lottery-ticket-hypothesis/blob/main/source/lth/models/__init__.py
class Layer:
    """Represents a single prunable layer in the neural network."""

    def __init__(
            self,
            name: str,
            weights: torch.nn.Parameter,
            initial_weights: torch.Tensor,
            pruning_mask: torch.Tensor) -> None:
        """Initializes a new Layer instance.

        Args:
            name (str): The name of the layer.
            kind (LayerKind): The kind of the layer.
            weights (torch.nn.Parameter): The weights of the layer.
            biases (torch.nn.Parameter): The biases of the layer.
            initial_weights (torch.Tensor): A copy of the initial weights of the layer.
            initial_biases (torch.Tensor): A copy of the initial biases of the layer.
            pruning_mask (torch.Tensor): The current pruning mask of the layer.
        """

        self.name = name
      
        self.weights = weights
        self.initial_weights = initial_weights
        self.pruning_mask = pruning_mask

# referenced from: https://github.com/lecode-official/pytorch-lottery-ticket-hypothesis/blob/main/source/lth/models/__init__.py
class BaseModel(torch.nn.Module):
    """Represents the base class for all models."""

    def __init__(self) -> None:
        """Initializes a new BaseModel instance. Since this is a base class, it should never be called directly."""

        # Invokes the constructor of the base class
        super().__init__()

        # Initializes some class members
        self.layers = None
        

    def initialize(self, device) -> None:
        """Initializes the model. It initializes the weights of the model using Xavier Normal (equivalent to Gaussian Glorot used in the original
        Lottery Ticket Hypothesis paper). It also creates an initial pruning mask for the layers of the model. These are initialized with all ones. A
        pruning mask with all ones does nothing. This method must be called by all sub-classes at the end of their constructor.
        """
        

        # Gets the all the fully-connected and convolutional layers of the model (these are the only ones that are being used right now, if new layer
        # types are introduced, then they have to be added here, but right now all models only consist of these two types)
        self.layers = []
        for parameter_name, parameter in self.named_parameters():
            weights = parameter
            
            weights.requires_grad = True
            init_weights=parameter.clone()
                
            # Initializes the pruning masks of the layer, which are used for pruning as well as freezing the pruned weights during training
            pruning_mask = torch.ones_like(init_weights, dtype=torch.uint8)
            pruning_mask.to(device)  # pylint: disable=no-member
            # Adds the layer to the internal list of layers
         
        
            self.layers.append(Layer(parameter_name, weights, init_weights, pruning_mask))
        
            

    def get_layer_names(self):
        """Retrieves the internal names of all the layers of the model.

        Returns:
            list[str]: Returns a list of all the names of the layers of the model.
        """

        layer_names = []
        for layer in self.layers:
            layer_names.append(layer.name)
        return layer_names

    def get_layer(self, layer_name: str) -> Layer:
        """Retrieves the layer of the model with the specified name.

        Args:
            layer_name (str): The name of the layer that is to be retrieved.

        Raises:
            LookupError: If the layer does not exist, an exception is raised.

        Returns:
            Layer: Returns the layer with the specified name.
        """

        for layer in self.layers:
            if layer.name == layer_name:
                return layer
        raise LookupError(f'The specified layer "{layer_name}" does not exist.')

    def update_layer_weights(self, mask, layer_name: str, new_weights: torch.Tensor) -> None:
        """Updates the weights of the specified layer.

        Args:
            layer_name (str): The name of the layer whose weights are to be updated.
            new_weights (torch.Tensor): The new weights of the layer.
        """

        
        with torch.no_grad():
            # Update the layer weights
            self.state_dict()[layer_name].copy_(new_weights)
            self.get_layer(layer_name).weights.copy_(new_weights)
    
            self.get_layer(layer_name).pruning_mask.copy_(mask)
        
    def get_total_num_weights(self):
        terms =0
        for l in self.layers:
            l = self.get_layer(l.name)
            terms += l.weights.flatten().shape[0]
        return terms

    def reset(self) -> None:
        """Resets the model back to its initial initialization."""

        for layer in self.layers:
            self.state_dict()[f'{layer.name}.weight'].copy_(layer.initial_weights)

    def move_to_device(self, device: Union[int, str, torch.device]) -> None:  # pylint: disable=no-member
        """Moves the model to the specified device.

        Args:
            device (Union[int, str, torch.device]): The device that the model is to be moved to.
        """

        # Moves the model itself to the device
        self.to(device)

        # Moves the initial weights, initial biases, and the pruning masks also to the device
        for layer in self.layers:
            layer.initial_weights = layer.initial_weights.to(device)
            layer.pruning_mask = layer.pruning_mask.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the neural network. Since this is the base model, the method is not implemented and must be implemented
        in all classes that derive from the base model.

        Args:
            x (torch.Tensor): The input to the neural network.

        Raises:
            NotImplementedError: _description_

        Returns:
            torch.Tensor: Returns the output of the neural network.
        """

        raise NotImplementedError()
class LLAMAEntailmentClassifier(BaseModel):
    def __init__(self, encoder_name="knowledgator/Llama-encoder-1.0B", vocab=None, freeze_encoder=False, device='cuda'):
        super().__init__()
        self.vocab = vocab
        self.encoder_name = encoder_name
        self.encoder = LlamaBiModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.tokenizer.model_max_length = 512
        
        if "pad_token" not in self.tokenizer.special_tokens_map:
            num_new_tokens = self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            num_new_tokens = 0
            
        if num_new_tokens > 0:
            self.encoder.resize_token_embeddings(len(self.tokenizer))
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.encoder_dim = 2048 #not sure how to get this through a code version
        self.mlp_input_dim = self.encoder_dim * 4
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(self.mlp_input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 3),
        )
        self.output_dim = 3
        self.initialize(device)

    def forward(self, s1, s1len, s2, s2len):
        device = s1.device
        
        s1 = s1.transpose(1, 0)
        s2 = s2.transpose(1, 0)
        
        s1_tokens = self.indices_to_bert_tokens(s1)
        s2_tokens = self.indices_to_bert_tokens(s2)
        
        s1_tokens = {k: v.to(device) for k, v in s1_tokens.items()}
        s2_tokens = {k: v.to(device) for k, v in s2_tokens.items()}
        
        s1enc = self.encode_sentence(s1_tokens)
        s2enc = self.encode_sentence(s2_tokens)
        
        diffs = s1enc - s2enc
        prods = s1enc * s2enc
        
        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)
        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)
        preds = self.mlp(mlp_input)
        
        return preds

    def get_final_reprs(self, s1, s1len, s2, s2len):
        device = s1.device
        
        s1 = s1.transpose(1, 0)
        s2 = s2.transpose(1, 0)
        
        s1_tokens = self.indices_to_bert_tokens(s1)
        s2_tokens = self.indices_to_bert_tokens(s2)
        
        s1_tokens = {k: v.to(device) for k, v in s1_tokens.items()}
        s2_tokens = {k: v.to(device) for k, v in s2_tokens.items()}
        
        s1enc = self.encode_sentence(s1_tokens)
        s2enc = self.encode_sentence(s2_tokens)
        
        diffs = s1enc - s2enc
        prods = s1enc * s2enc
        
        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)
        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)
        rep = self.mlp[:-1](mlp_input)
        
        return rep

    def forward_from_final(self, rep):
        preds = self.mlp[-1:](rep)
        return preds

    def indices_to_bert_tokens(self, indices):
        batch_size, seq_len = indices.shape
        words = []
        for i in range(batch_size):
            sentence = []
            for idx in indices[i]:
                if idx.item() in self.vocab['itos']:
                    word = self.vocab['itos'][idx.item()]
                    if word not in ("[PAD]", "<pad>", "PAD"): 
                        sentence.append(word)
                else:
                    break
            words.append(sentence)
        
        return self.tokenizer(words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)

    def encode_sentence(self, tokens):
        outputs = self.encoder(**tokens).last_hidden_state.mean(dim=1) 
        return outputs

    def to(self, device):
        self.encoder = self.encoder.to(device)
        return super().to(device)


class BertEntailmentClassifier(BaseModel):
    def __init__(self, encoder_name="bert-base-uncased", vocab=None, freeze_bert=False, device='cuda'):
        super().__init__()
        self.vocab = vocab
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)

#         if freeze_bert:
#             for param in self.encoder.parameters():
#                 param.requires_grad = False

        self.encoder_dim = self.encoder.config.hidden_size
        self.mlp_input_dim = self.encoder_dim * 4
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(self.mlp_input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 3),
        )
        self.output_dim = 3
        self.initialize(device)
        

    def forward(self, s1, s1len, s2, s2len):
        device = s1.device
        
        s1 = s1.transpose(1, 0)
        s2 = s2.transpose(1, 0)
        
        s1_tokens = self.indices_to_bert_tokens(s1)
        s2_tokens = self.indices_to_bert_tokens(s2)
        
        s1_tokens = {k: v.to(device) for k, v in s1_tokens.items()}
        s2_tokens = {k: v.to(device) for k, v in s2_tokens.items()}
        
        s1enc = self.encode_sentence(s1_tokens)
        s2enc = self.encode_sentence(s2_tokens)
        
        diffs = s1enc - s2enc
        prods = s1enc * s2enc
        
        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)
        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)
        preds = self.mlp(mlp_input)
        
        return preds

    def get_final_reprs(self, s1, s1len, s2, s2len):
        device = s1.device
        
        s1 = s1.transpose(1, 0)
        s2 = s2.transpose(1, 0)
        
        s1_tokens = self.indices_to_bert_tokens(s1)
        s2_tokens = self.indices_to_bert_tokens(s2)
        
        s1_tokens = {k: v.to(device) for k, v in s1_tokens.items()}
        s2_tokens = {k: v.to(device) for k, v in s2_tokens.items()}
        
        s1enc = self.encode_sentence(s1_tokens)
        s2enc = self.encode_sentence(s2_tokens)
        
        diffs = s1enc - s2enc
        prods = s1enc * s2enc
        
        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)
        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)
        rep = self.mlp[:-1](mlp_input)
        
        return rep

    def forward_from_final(self, rep):
        preds = self.mlp[-1:](rep)
        return preds

    def indices_to_bert_tokens(self, indices):
        batch_size, seq_len = indices.shape
        words = []
        for i in range(batch_size):
            sentence = []
            for idx in indices[i]:
                if idx.item() in self.vocab['itos']:
                    word = self.vocab['itos'][idx.item()]
                    if word not in ("[PAD]", "<pad>", "PAD"): 
                        sentence.append(word)
                else:
                    break
            words.append(sentence)
        
        return self.tokenizer(words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)

    def encode_sentence(self, tokens):
        outputs = self.encoder(**tokens)
        return outputs.last_hidden_state[:, 0, :]

    def to(self, device):
        self.encoder = self.encoder.to(device)
        return super().to(device)


class BowmanEntailmentClassifier(BaseModel):
    """
    The RNN-based entailment model of Bowman et al 2017
    """

    def __init__(self, encoder, device):
        super().__init__()

        self.encoder = encoder
        self.encoder_dim = encoder.output_dim
        self.mlp_input_dim = self.encoder_dim * 4
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(self.mlp_input_dim)
        self.prune_mask= torch.ones(1024,self.mlp_input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),  # Mimic classifier MLP keep rate of 94%
            nn.Linear(1024, 3),
        )
        #self.mlp[:-1][0] = prune.ln_structured(self.mlp[:-1][0], name="weight", amount=0.05, dim=1, n=float('-inf'))
        self.output_dim = 3
        
        self.initialize(device)
        
       
        
        
    def forward(self, s1, s1len, s2, s2len):
        s1enc = self.encoder(s1, s1len)
        s2enc = self.encoder(s2, s2len)

        diffs = s1enc - s2enc
        prods = s1enc * s2enc

        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1) #1x2048
    
        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)
        
        preds = self.mlp(mlp_input)

        return preds
    
    
        
   
            
    
        
    # from https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/lottery/utils.py
    def copy_weights_linear(linear_unpruned, linear_pruned):
        """Copy weights from an unpruned model to a pruned model.

        Modifies `linear_pruned` in place.

        Parameters
        ----------
        linear_unpruned : nn.Linear
            Linear model with a bias that was not pruned.

        linear_pruned : nn.Linear
            Linear model with a bias that was pruned.
        """
        assert check_pruned_linear(linear_pruned)
        assert not check_pruned_linear(linear_unpruned)

        with torch.no_grad():
            linear_pruned.weight_orig.copy_(linear_unpruned.weight)
            linear_pruned.bias_orig.copy_(linear_unpruned.bias)

    def get_final_reprs(self, s1, s1len, s2, s2len):
        s1enc = self.encoder(s1, s1len)
        s2enc = self.encoder(s2, s2len)

        diffs = s1enc - s2enc
        prods = s1enc * s2enc

        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)

        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)
        
                
        rep = self.mlp[:-1](mlp_input) 
        
        return rep

    def forward_from_final(self, rep):
        preds = self.mlp[-1:](rep)
        return preds
    
    def get_encoder(self):
        return self.encoder

    



class DropoutLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.W_i = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.U_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        self.W_f = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.U_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        self.W_c = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.U_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        self.W_o = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self._input_dropout_mask = self._h_dropout_mask = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.W_i)
        nn.init.orthogonal_(self.U_i)
        nn.init.orthogonal_(self.W_f)
        nn.init.orthogonal_(self.U_f)
        nn.init.orthogonal_(self.W_o)
        nn.init.orthogonal_(self.U_o)
        nn.init.orthogonal_(self.W_c)
        nn.init.orthogonal_(self.U_c)
        self.b_f.data.fill_(1.0)
        self.b_i.data.fill_(1.0)
        self.b_o.data.fill_(1.0)

    def set_dropout_masks(self, batch_size):
        if self.dropout:
            if self.training:
                self._input_dropout_mask = torch.bernoulli(
                    torch.Tensor(4, batch_size, self.input_size).fill_(1 - self.dropout)
                )
                self._input_dropout_mask.requires_grad = False
                self._h_dropout_mask = torch.bernoulli(
                    torch.Tensor(4, batch_size, self.hidden_size).fill_(
                        1 - self.dropout
                    )
                )
                self._h_dropout_mask.requires_grad = False

                if torch.cuda.is_available():
                    self._input_dropout_mask = self._input_dropout_mask.cuda()
                    self._h_dropout_mask = self._h_dropout_mask.cuda()
            else:
                self._input_dropout_mask = self._h_dropout_mask = [
                    1.0 - self.dropout
                ] * 4
        else:
            self._input_dropout_mask = self._h_dropout_mask = [1.0] * 4

    def forward(self, input, hidden_state):
        h_tm1, c_tm1 = hidden_state

        if self._input_dropout_mask is None:
            self.set_dropout_masks(input.size(0))

        xi_t = F.linear(input * self._input_dropout_mask[0], self.W_i, self.b_i)
        xf_t = F.linear(input * self._input_dropout_mask[1], self.W_f, self.b_f)
        xc_t = F.linear(input * self._input_dropout_mask[2], self.W_c, self.b_c)
        xo_t = F.linear(input * self._input_dropout_mask[3], self.W_o, self.b_o)

        i_t = F.sigmoid(xi_t + F.linear(h_tm1 * self._h_dropout_mask[0], self.U_i))
        f_t = F.sigmoid(xf_t + F.linear(h_tm1 * self._h_dropout_mask[1], self.U_f))
        c_t = f_t * c_tm1 + i_t * F.tanh(
            xc_t + F.linear(h_tm1 * self._h_dropout_mask[2], self.U_c)
        )
        o_t = F.sigmoid(xo_t + F.linear(h_tm1 * self._h_dropout_mask[3], self.U_o))
        h_t = o_t * F.tanh(c_t)

        return h_t, c_t



class TextEncoder(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim=300, hidden_dim=512, bidirectional=False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.bidirectional = bidirectional
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=1)
        self.rnn = nn.LSTM(
            self.embedding_dim, self.hidden_dim, bidirectional=bidirectional
        )
        self.output_dim = self.hidden_dim

    def forward(self, s, slen):
        semb = self.emb(s)
        spk = pack_padded_sequence(semb, slen.cpu(), enforce_sorted=False)
        _, (hidden, cell) = self.rnn(spk)
        #retunr get all cell states w a param for the cell state # 
        return hidden[-1]
        

    def get_states(self, s, slen):
        semb = self.emb(s)
        spk = pack_padded_sequence(semb, slen.cpu(), enforce_sorted=False)
        outputs, _ = self.rnn(spk)
        outputs_pad = pad_packed_sequence(outputs)[0]
        return outputs_pad #padded hidden states for each word
    
    def get_last_cell_state(self, s,slen):
        semb = self.emb(s)
        spk = pack_padded_sequence(semb, slen.cpu(), enforce_sorted=False)
        _, (hidden, cell) = self.rnn(spk)
        
        
        return cell[-1]


class DropoutTextEncoder(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim=300, hidden_dim=512, bidirectional=False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.bidirectional = bidirectional
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=1)
        self.rnn_cell = DropoutLSTMCell(
            self.embedding_dim, self.hidden_dim, dropout=0.5
        )
        self.output_dim = self.hidden_dim

    def forward(self, s, slen):
        semb = self.emb(s)

        hx = torch.zeros(semb.shape[1], self.hidden_dim).to(semb.device)
        cx = torch.zeros(semb.shape[1], self.hidden_dim).to(semb.device)
        for i in range(semb.shape[0]):
            hx, cx = self.rnn_cell(semb[i], (hx, cx))
        return hx

    def get_states(self, s, slen):
        raise NotImplementedError
