#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/protobuf/config.pb.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/status.h>
#include <tensorflow/cc/framework/ops.h>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <re2/re2.h>
#include <re2/stringpiece.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <string_view>
#include <stdexcept>
#include <iomanip>

std::vector<std::string_view> splitString(std::string_view s, const re2::RE2& re) {
    std::vector<std::string_view> results;
    if (re.ok()) {
        re2::StringPiece input(s);
        re2::StringPiece s;
        while (RE2::FindAndConsume(&input, re, &s)) {
            results.push_back(s);
        }
    } else {
        std::cout << "Not a valid regular expression.\n";
    }
    return results;
}

tensorflow::Tensor vector_to_tensor_2d(const std::vector<std::vector<float>>& data) {
  const int M = data.size();
  const int N = M > 0 ? data[0].size() : 0;
  tensorflow::Tensor input(tensorflow::DT_FLOAT, tensorflow::TensorShape({M, N}));
  auto input_map = input.tensor<float, 2>();
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      input_map(i, j) = data[i][j];
    }
  }
  return input;
}

auto read_gzipped_csv(const std::string& filename) {
  std::ifstream ifs_compressd_in(filename.c_str());
  boost::iostreams::filtering_istream in;
  in.push(boost::iostreams::gzip_decompressor());
  in.push(ifs_compressd_in);
  std::vector<std::vector<float>> data;
  std::string line;
  bool first_line = true;
  const re2::RE2 re("([^,]+)");
  while (std::getline(in, line)) {
    if (first_line) {
      first_line = false;
      continue;
    } else if (line.empty()) {
      continue;
    } else {
      std::vector<float> fields;
      std::vector<std::string_view> tmp_fields = splitString(line, re);
      bool first_field = true;
      for (const auto& word: tmp_fields) {
        if (first_field) {
          first_field = false;
          continue;
        }
        fields.push_back(std::stod(word.data()));
      }
      data.push_back(std::move(fields));
    }
  }
  return data;
}

void write_2d_tensor_to_file(const std::string& output_filename, const tensorflow::Tensor& tensor) {
  const tensorflow::TensorShape shape = tensor.shape();
  if (shape.dims() != 2) {
    throw std::runtime_error("Expect a 2D tensor.");
  }
  const int M = shape.dim_size(0);
  const int N = shape.dim_size(1);
  auto output_map = tensor.matrix<float>();
  std::ofstream ofs(output_filename);
  if (ofs) {
    for (int j = 0; j < N - 1; ++j) {
      ofs << "CV" + std::to_string(j + 1) << ",";
    }
    ofs << "CV" + std::to_string(N) << '\n';
    ofs << std::setprecision(12);
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N - 1; ++j) {
        ofs << output_map(i, j) << ",";
      }
      ofs << output_map(i, N-1);
      ofs << '\n';
    }
  } else {
    throw std::runtime_error("Failed to write to stream");
  }
}

int main() {
  const std::string model_dir{"best_encoder_model/"};
  tensorflow::SavedModelBundleLite bundle;
  auto status = tensorflow::LoadSavedModel(
    tensorflow::SessionOptions(), tensorflow::RunOptions(), model_dir,
    {tensorflow::kSavedModelTagServe}, &bundle);
  if (status.ok()) {
    std::cout << "Load OK\n";
    const auto input_tensor = vector_to_tensor_2d(read_gzipped_csv("dx0.1_dy1.0.csv.gz"));
    const auto session = bundle.GetSession();
    std::vector<tensorflow::Tensor> output;
    status = session->Run(
      {{"serving_default_encoder_layer_0:0", input_tensor}},
      {"StatefulPartitionedCall:0"}, {}, &output);
    if (status.ok()) {
      std::cout << "Compute OK\n";
      write_2d_tensor_to_file("encoded_cpp.csv", output[0]);
    } else {
      std::cout << "Compute error: " << status.error_message() << "\n";
    }
  } else {
    std::cout << "Load error: " << status.error_message() << "\n";
  }
  return 0;
}
