$ source /home/james/anaconda3/bin/activate
/bin/sh: 1: source: not found
$ conda activate pytorch3copy
/bin/sh: 2: conda: not found
$ /home/james/anaconda3/envs/pytorch3copy/bin/python /home/james/VSCode/currentPipelines/training.py
torch version: 1.10.0, ignite version: 0.4.7
Current working directory: /home/james/VSCode/currentPipelines
torch cuda current device: 0
Memory/bytes allocated after model instantiation: 43318272
Memory/bytes allocated after ronchdset instantiation: 43318272
Total number of Ronchigrams used: 100002
trainFraction:evalFraction:testFraction 0.7:0.15:0.15000000000000005
Memory/bytes allocated after ronchdset splitting: 43318272
Memory/bytes allocated after creating data loaders: 43318272
Experiment name:  20220218-121946
Epoch [1/4]:  25%|███████████████████████████                                                                                 | 1/4 [00:00<?, ?it/s]/home/james/anaconda3/envs/pytorch3copy/lib/python3.9/site-packages/ignite/contrib/handlers/base_logger.py:96: UserWarning: Provided metric name 'batchloss' is missing in engine's state metrics: []
  warnings.warn(
                                                                                                                                                   /home/james/anaconda3/envs/pytorch3copy/lib/python3.9/site-packages/pytorch_forecasting/metrics.py:555: UserWarning: Loss is not finite. Resetting it to 1e9
  warnings.warn("Loss is not finite. Resetting it to 1e9")
Current run is terminating due to exception: element 0 of tensors does not require grad and does not have a grad_fn
Engine run is terminating due to exception: element 0 of tensors does not require grad and does not have a grad_fn
Traceback (most recent call last):
  File "/home/james/VSCode/currentPipelines/training.py", line 581, in <module>
    trainer.run(trainLoader, max_epochs=num_epochs)
  File "/home/james/anaconda3/envs/pytorch3copy/lib/python3.9/site-packages/ignite/engine/engine.py", line 698, in run
    return self._internal_run()
  File "/home/james/anaconda3/envs/pytorch3copy/lib/python3.9/site-packages/ignite/engine/engine.py", line 771, in _internal_run
    self._handle_exception(e)
  File "/home/james/anaconda3/envs/pytorch3copy/lib/python3.9/site-packages/ignite/engine/engine.py", line 466, in _handle_exception
    raise e
  File "/home/james/anaconda3/envs/pytorch3copy/lib/python3.9/site-packages/ignite/engine/engine.py", line 741, in _internal_run
    time_taken = self._run_once_on_dataset()
  File "/home/james/anaconda3/envs/pytorch3copy/lib/python3.9/site-packages/ignite/engine/engine.py", line 845, in _run_once_on_dataset
    self._handle_exception(e)
  File "/home/james/anaconda3/envs/pytorch3copy/lib/python3.9/site-packages/ignite/engine/engine.py", line 466, in _handle_exception
    raise e
  File "/home/james/anaconda3/envs/pytorch3copy/lib/python3.9/site-packages/ignite/engine/engine.py", line 831, in _run_once_on_dataset
    self.state.output = self._process_function(self, self.state.batch)
  File "/home/james/VSCode/currentPipelines/training.py", line 388, in update_fn
    loss.backward()
  File "/home/james/anaconda3/envs/pytorch3copy/lib/python3.9/site-packages/torch/_tensor.py", line 307, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
  File "/home/james/anaconda3/envs/pytorch3copy/lib/python3.9/site-packages/torch/autograd/__init__.py", line 154, in backward
    Variable._execution_engine.run_backward(
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
$ 