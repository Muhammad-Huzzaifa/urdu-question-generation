| | |
|---|---|
| Encoder / decoder type (LSTM or GRU) | LSTM / LSTM |
| Layers / embedding / hidden size | 2 / 256 / 512 |
| Vocabulary size | 8000 |
| Trainable parameters | 21,645,120 |
| Optimiser, learning rate, schedule | Adam, lr=0.001, weight_decay=0.0001, ReduceLROnPlateau(factor=0.5, patience=1) |
| Batch size, epochs, wall-clock, GPU | 64, 15, 1:37:00, Tesla T4 |
