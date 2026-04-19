# cd_trainer
Performed the following changes

### Huber loss 
Change primary cd loss function from MSE to Huber to reduce exploding losses.
MSE squares huge errors while Huber acts like L2 for small errors and L1 for massive errors.

### Fix timestep tensor size 
Was passing in 0-dim tensor for student prediction which could be broadcast incorrectly.
Now it is a 1-dim tensor based on the batch size

### Clip gradients
Bad guesses result in big losses which can explode the gradient.
Clipping should preserve direction but scale down magnitude
Chose `max=1.0` because it is the standard max to clip gradients.

### Anchor loss
Because student initialized with teacher's weights it predicts a local step from that point. 
However, the goal is a global step (remove all noise immediately) which requires context from the teacher's outputs. 
This results in the student's tiny errors on a local scale to explode to a global scale.
Add the teacher anchor loss to weigh the local error less heavily.  

### Transfer `alpha_t` from consistency.py to GPU
`alpha_t` might default to the CPU so keeping everything on GPU prevents silent clipping 
