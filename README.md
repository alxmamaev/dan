# DAN
Deep Averaging Networks<br>
[[Paper](https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf)]


## Example
```python
import torch
from dan import DAN

vocab_size = 100000
vector_size = 300
num_classes = 5

batch_size = 32
sequence_len = 100

dan = DAN(vocab_size, vector_size, num_classes)
input_data = torch.randint(0, vocab_size, (batch_size, sequence_len))
output = dan(input_data)

print(output)
```
