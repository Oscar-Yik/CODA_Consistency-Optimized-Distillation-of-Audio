# cd_trainer
Performed the following changes

### Huber loss 
Change primary cd loss function from MSE to Huber to reduce exploding losses.
MSE squares huge errors while Huber acts like L2 for small errors and L1 for massive errors.

### Anchor loss
Because student initialized with teacher's weights it predicts a local step from that point. 
However, the goal is a global step (remove all noise immediately) which requires context from the teacher's outputs. 
This results in the student's tiny errors on a local scale to explode to a global scale.
Add the teacher anchor loss to weigh the local error less heavily.  

### Gradient Clipping
Remove gradient clipping in `models/consistency.py` but introduce it in `cd_trainer.py`. 
Apprently this allows the model to learn from examples with a lot of noise but reduces gradient explosions at the same time. 

# train_consistency

### Remove curriculum 
Apparently the curriculum allows the model to cheat? Also randomness is better for lots of data?

# Next things to try
Adaptive EMA decay: start at 0.95 and slowly increase to 0.999
Conditioning Dropout: set mean content to completly blanks arrays 10% of the time so UNet learns mathematical relatonship between f0 pitch contour and target voice
