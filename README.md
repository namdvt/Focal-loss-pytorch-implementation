## Project Title

An Pytorch implementation of the focal loss which proposed in: 
"Focal Loss for Dense Object Detection", https://arxiv.org/abs/1708.02002
## Example

Training
```
import function as F

optimizer.zero_grad()
criterion = F.FocalLoss()

output = model(data)

loss = criterion(output, target)
loss.backward()
optimizer.step()
```