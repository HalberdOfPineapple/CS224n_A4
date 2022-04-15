# Assignment 4 Notes

## Part I: Model Building

### (b)

```python
self.encoder = nn.LSTM(embed_size, hidden_size, bias=True, bidirectional=True)
self.decoder = nn.LSTMCell(embed_size + hidden_size, hidden_size, True)
```

- the difference between `nn.LSTM` and `nn.LSTMCell`
  - `nn.LSTM` will take inputs in shape `(seq_len, batch_size, embedding_size)` (or `batch_size` at first if `batch_first` = True)
    - where `embedding_size` is the input size in the documentation
    - and output an output tensor with shape `(seq_len, batch_size, embedding_size)`
      - if `batch_first = True` , batch size will be put in the first
      - if `bidirectional = True` , the output vector dimension will be 2 * hidden size
  - while `nn.LSTMCell` can be treated as a single step in the LSTM
    - it will take an input with shape `(batch_size, input_size)` and (hidden state, cell state)
    - and output the next hidden state and cell state

### (c)

```python
self.source = nn.Embedding(len(vocab.src), self.embed_size, src_pad_token_idx)
self.target = nn.Embedding(len(vocab.tgt), self.embed_size, tgt_pad_token_idx)
```

- `torch.nn.Embedding` 
  - `num_embeddings` - here refers to the vocabulary size
  - `embedding_dim` - refers to the dimensionality of embedded vectors
  - `padding_idx` - the index of padding token in the vocabulary

### (d)

General procedure:

- `source_paddded` fed into embedding layer to convert it to be (batch of) sequences of word embeddings X

  - `(src_len, b, e)` where b is the batch size
  - without setting `batch_first` to be True, the batch dimension will be in the middle

- `X` is to be packed using function `pack_padded_sequence` for using their actual sequence lengths when encoding

  - ```python
    X = pack_padded_sequence(X, torch.tensor(source_lengths))
    ```

  - this resulted X is a `PackedSequence` object and all RNN modules accept such instances as input

- pass packed input to the encoder and pad the packed output:

  - ```python
    enc_hiddens, (last_hidden, last_cell) = self.encoder(X)
    enc_hiddens, _ = pad_packed_sequence(enc_hiddens, batch_first=True)
    ```

  - `pad_packed_sequence` is the reverse operation of `pack_padded_sequence` 

    - its output is a tuple with the second one being tensor of lengths
    - `bach_first` here is because required shape takes batch_size as the first `(b, src_len, e)`

  - the LSTM (encoder) will also **output the last hidden state and last cell state as a tuple**

    - their shape are `(layer_num, batch_size, hidden_Size` )

- Concatenate the last hidden states of 2 directions (and also for cells) and project them to be `(b, hidden_size)`

  - ```python
    init_decoder_hidden = self.h_projection(torch.cat((last_hidden[0], last_hidden[1]),dim=1))
    init_decoder_cell = self.c_projection(torch.cat((last_cell[0], last_cell[1]), dim=1))
    ```

### (e) & (f)

for iterating along first dimension of `Y` which has shape `(tgt_len, b, e)`

```python
for Y_t in torch.split(Y, 1):
```

- `torch.split` 's second argument (if it is an integer) specifies the stride iterating on specified dimension `dim`
- return a tuple of split tensors
- here it will return the corresponding (batch of) embedding vectors with shape `(b, e)`



for obtaining the next decoder states using the single `LSTMCell`

```python
dec_state = self.decoder(Ybar_t, dec_state)
```

- where the input to `LSTMCell` has shape `(batch_size, input_size`) . the input size is specified when creating

for doing batched matrix multiplication:

```python
e_t = torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_state[0], dim=2))
e_t = torch.squeeze(e_t, dim=2)
```

- `torch.bmm` only takes 2 **3-d tensors** and ignores the first dimension when doing the matrix multiplication (ignoring batch dimension)
- `torch.unsqueeze(input, dim)` will insert 1 in the specified dimension
  - here `torch.unsqueeze(dec_state[0], dim=2)` will transform it from `(b, h)` to be `(b,h,1)` 
- `torch.squeeze` is the reverse operation of `unsqueeze`. It will remove 1 in specified dimenison:
  - here `troch.squeeze(e_t, dim=2)` will transform it from `(b, src_len, 1)` to be `(b,src_len)`

