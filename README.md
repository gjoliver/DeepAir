# DeepAir
Demonstrates how to run a minimal DeepSpeed job with Ray AIR's TorchTrainer

Main entry-point is torch_trainer.py

```
python torch_trainer.py
```

There is also ```core.py```, which runs DeepSpeed on a Ray cluster using Ray Core and Ray Core only.

This is mostly for demonstrating what is happening behind the scene.
Running DeepSpeed this way is not recommeneded, since you won't get all the benefits of fault tolerance and retries offered by Ray AIR.

```
python core.py
```
