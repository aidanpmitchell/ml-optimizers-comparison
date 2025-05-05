
# Regular Gradient Descent Implementation
class GradientDescent:
    # default learning rate is 0.01
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                # update rule
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_() # reset gradients to zero

