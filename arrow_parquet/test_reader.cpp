#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/visitor.h>
#include <parquet/arrow/reader.h>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <arrow/array.h>
#include <arrow/scalar.h>
#include <iostream>
#include <fmt/format.h>

class SumVisitor: public arrow::ScalarVisitor {
public:
  SumVisitor() {m_sum = 0;}
  arrow::Status Visit(const arrow::Int64Scalar &scalar) override {
    m_sum += int64_t(static_cast<arrow::Int64Scalar>(scalar).value);
    return arrow::Status::OK();
  }
  arrow::Status Visit(const arrow::DoubleScalar &scalar) override {
    m_sum += double(static_cast<arrow::DoubleScalar>(scalar).value);
    return arrow::Status::OK();
  }
  double getResult() const {return m_sum;}
private:
  double m_sum;
};

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Please provide at least one argument." << std::endl;
    return -1;
  }
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::io::RandomAccessFile> input;

  auto result = arrow::io::ReadableFile::Open(argv[1]);
  if (!result.ok()) {
    std::cerr << "Cannot open " << argv[1] << std::endl;
    return -1;
  }
  input = result.ValueOrDie();

  // Open Parquet file reader
  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  auto status = parquet::arrow::OpenFile(input, pool, &arrow_reader);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return -1;
  }

  std::shared_ptr<arrow::Table> table;
  status = arrow_reader->ReadTable(&table);
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    return -1;
  }

  // Read columns
  auto column_names = table->ColumnNames();
  std::cout << "Number of fields: " << column_names.size() << std::endl;
  const auto available_types = arrow::PrimitiveTypes();
  for (size_t i = 0; i < column_names.size(); ++i) {
    auto current_column = table->column(i);
    const auto num_chunks = current_column->num_chunks();
    std::cout << fmt::format("Column {}, name {}, {} chunk(s)", i, column_names[i], num_chunks);
    SumVisitor v;
    for (int j = 0; j < num_chunks; ++j) {
      std::cout << fmt::format(", chunk {} data: [", j);
      auto current_chunk = current_column->chunk(j);
      // Print data and sum the values
      for (int64_t k = 0; k < current_chunk->length(); ++k) {
        auto current_scalar = current_chunk->GetScalar(k).ValueOrDie();
        std::cout << " " << current_scalar->ToString();
        const auto result = current_scalar->Accept(&v);
      }
      std::cout << " ]";
      // Print types
      std::cout << ", type: [";
      for (int64_t k = 0; k < current_chunk->length(); ++k) {
        auto current_scalar = current_chunk->GetScalar(k).ValueOrDie();
        std::cout << " " << current_scalar->type->name();
      }
      std::cout << " ], sum = " << v.getResult();
    }
    std::cout << std::endl;
  }
  return 0;
}
