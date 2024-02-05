## Here will be the list of all found and corrected errors:


### modeling/diffusion.py

1. time should be put to the same device as x in self.forward()
2. random noise should be put on device before eps_model, not after 

### modeling/training.py


### modeling/unet.py

1Size of time_embedding should match the size of tensor it is added to. So we do [:, :, None, None]


### tests/test_model.py

1. Test for diffusion loss is really strange, we should not expect it to be between 1.0 and 1.2. 
We, though, anticipate it to be non-negative.