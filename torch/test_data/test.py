#!/usr/bin/env python3
import torch
import numpy as np


encoder_model = torch.jit.load('srvs_10_100_test_2_10_best_encoder_model.pt')
encoder_model.to('cpu')
encoder_input = torch.tensor(np.array([[0.5, 0.6], [-0.6, 0.6]]), dtype=torch.float32, requires_grad=True)
encoder_output = encoder_model(encoder_input)
print(encoder_output)
y = encoder_output
print(next(encoder_model.parameters()).dtype)

encoder_input_derivative = torch.ones_like(y)
grad = torch.autograd.grad(outputs=y, inputs=encoder_input, grad_outputs=encoder_input_derivative)
print(grad)

jacobian = torch.autograd.functional.jacobian(encoder_model, encoder_input)
print(jacobian)
