#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <stdexcept>

void print_tensor(const at::Tensor& tensor) {
  const auto tensor_dim = tensor.sizes();
  std::vector<decltype(tensor_dim)::value_type> index(tensor_dim.size(), 0);
  for (size_t i = 0; i < tensor_dim.size(); ++i) {
    for (int64_t j = static_cast<int64_t>(tensor_dim.size()) - 1; j >= i; --j) {
      index[j] = 0;
      for (int64_t k = j; k < tensor_dim.size(); ++k) {
        index[k] += 1;
      }
    }
  }
}

auto numerical_gradient(torch::jit::script::Module& module, std::array<float, 2> input_data) {
  auto input_tensor = torch::from_blob(
      input_data.data(), {2}, torch::TensorOptions().dtype(c10::CppTypeToScalarType<float>::value));
  input_tensor.set_requires_grad(true);
  const std::vector<torch::jit::IValue> input_args{input_tensor};
  const auto output_tensor = module.forward(input_args).toTensor();
  std::array<float, 2> input_derivatie{0, 0};
  const float eps = 0.001f;
  std::cout << "Numerical gradients:\n";
  std::cout << output_tensor.size(0) << std::endl;
  for (size_t i = 0; i < input_data.size(); ++i) {
    const auto saved = input_data[i];
    input_data[i] = saved + eps;
    auto output_tensor_next = module.forward(input_args).toTensor();
    input_data[i] = saved - eps;
    auto output_tensor_prev = module.forward(input_args).toTensor();
    const auto output_value_next = torch::sum(output_tensor_next).item().to<float>();
    const auto output_value_prev = torch::sum(output_tensor_prev).item().to<float>();
    input_derivatie[i] = (output_value_next - output_value_prev) / (2.0f * eps);
    std::cout << input_derivatie[i] << '\n';
    input_data[i] = saved;
  }
  return input_derivatie;
}

std::vector<std::vector<float>> numerical_jacobian(torch::jit::script::Module& module, std::array<float, 2> input_data) {
  std::cout << "Numerical jacobian:\n";
  auto input_tensor = torch::from_blob(
      input_data.data(), {2}, torch::TensorOptions().dtype(c10::CppTypeToScalarType<float>::value));
  input_tensor.set_requires_grad(true);
  const std::vector<torch::jit::IValue> input_args{input_tensor};
  const auto output_tensor = module.forward(input_args).toTensor();
  const int num_output_units = output_tensor.size(0);
  std::vector<std::vector<float>> jacobian(num_output_units, std::vector<float>(input_data.size(), 0));
  const float eps = 0.001f;
  for (size_t i = 0; i < input_data.size(); ++i) {
    const auto saved = input_data[i];
    input_data[i] = saved + eps;
    auto output_tensor_next = module.forward(input_args).toTensor();
    input_data[i] = saved - eps;
    auto output_tensor_prev = module.forward(input_args).toTensor();
    for (int j = 0; j < num_output_units; ++j) {
      jacobian[j][i] = (output_tensor_next[j] - output_tensor_prev[j]).item().to<float>() / (2.0f * eps);
      std::cout << "Jacobian: output " << j << " to input " << i << " = " << jacobian[j][i] << std::endl;
    }
  }
  return jacobian;
}

auto torch_jacobian(torch::jit::script::Module& module, std::array<float, 2> input_data) {
  // TODO
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
    module.to(at::Device("cpu"));
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  auto it = module.parameters().begin();
  const auto module_input_dtype = (*it).dtype().toScalarType();
  std::cout << module_input_dtype << std::endl;
  std::vector<torch::jit::IValue> input_args(1);
  if (c10::CppTypeToScalarType<float>::value != module_input_dtype) {
    throw std::runtime_error("Different types!");
  }
  std::array<float, 2> input_data = {0.5, 0.6};
  auto input_tensor = torch::from_blob(
      input_data.data(), {input_data.size()}, torch::TensorOptions().dtype(module_input_dtype));
  input_tensor.set_requires_grad(true);
  input_args[0] = input_tensor;
  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(input_args).toTensor();
  const auto input_derivatives = torch::ones_like(output);
  std::cout << "Input data:\n";
  for (const auto& x: input_data) std::cout << x << " ";
  std::cout << "\n";
  std::cout << "Output:\n" << output << '\n';
  auto gradient = torch::autograd::grad({output}, {input_tensor}, {input_derivatives}, false);
  std::cout << "Gradient:\n" << gradient << std::endl;
//  std::cout << "Gradient2:\n" << input_tensor.grad() << std::endl;
  numerical_jacobian(module, input_data);
  std::cout << std::endl;
  input_data[0] = -0.6;
  output = module.forward(input_args).toTensor();
//  output.backward({}, true);
  std::cout << output << '\n';
  gradient = torch::autograd::grad({output}, {input_tensor}, {input_derivatives}, false);
  std::cout << "Gradient:\n" << gradient << std::endl;
//  std::cout << "Gradient2:\n" << input_tensor.grad() << std::endl;
  numerical_gradient(module, input_data);
  std::cout << "ok\n";
}
