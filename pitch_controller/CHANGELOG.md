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
