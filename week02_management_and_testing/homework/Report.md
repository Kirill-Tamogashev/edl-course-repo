# Report: how I searched for errors and bugs and corrected them. 

## Here is the list of all found and corrected errors and bugs:

### In `modeling/diffusion.py`

1. time should be put to the same device as x in self.forward()
2. random noise should be put on device before eps_model, not after. It was overall important
to make sure that any new tensor is created on a proper device

### In `modeling/training.py`
1. I wrote the combined train function, where all training is handled together.

### In `modeling/unet.py`

1. Size of time_embedding should match the size of tensor it is added to. So we do `[:, :, None, None]`


### In `tests/test_model.py`

1. Test for diffusion loss is really strange, we should not expect it to be between 1.0 and 1.2. 
We, though, anticipate it to be non-negative.


## How I found these bugs

1. First, I tried to run tests, if they failed, red the code and tried to understand if tests are correct
I though how I would write them

2. I debugged the networks, using `print()` as well, I checked the correctness of shapes, devices, verified 
absence of Nan.

3. I gradually assembled the pipeline, starting from UnetModel debug and finishing with integrated 
run of the training process, which included dataset and dataloader, logging (wandb mode="offline", to
avoid logging while debugging), optimizer setup, test step, train step. 