#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/type_fwd.h>
#include <arrow/io/file.h>
#include <iostream>
#include <parquet/arrow/writer.h>

int main() {
  // Some data to write
  std::vector<std::string> title{"X", "Y"};
  std::vector<double> data_x{0.5, 1.0, 2.6};
  std::vector<double> data_y{-10.0, 1.9, 12.7};

  // Build arrays
  arrow::DoubleBuilder builder_x;
  auto status = builder_x.AppendValues(data_x);
  if (!status.ok()) {
    std::cerr << status.message() << std::endl;
    return -1;
  }
  std::shared_ptr<arrow::Array> array_x = builder_x.Finish().ValueOrDie();

  arrow::DoubleBuilder builder_y;
  status = builder_y.AppendValues(data_y);
  if (!status.ok()) {
    std::cerr << status.message() << std::endl;
    return -1;
  }
  std::shared_ptr<arrow::Array> array_y = builder_y.Finish().ValueOrDie();

  // Create schema
  std::shared_ptr<arrow::Field> field_x = arrow::field(title[0], arrow::float64());
  std::shared_ptr<arrow::Field> field_y = arrow::field(title[1], arrow::float64());
  std::shared_ptr<arrow::Schema> schema = arrow::schema({field_x, field_y});

  arrow::ArrayVector vecs_x{array_x};
  std::shared_ptr<arrow::ChunkedArray> chunks_x = std::make_shared<arrow::ChunkedArray>(vecs_x);

  arrow::ArrayVector vecs_y{array_y};
  std::shared_ptr<arrow::ChunkedArray> chunks_y = std::make_shared<arrow::ChunkedArray>(vecs_y);

  std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {chunks_x, chunks_y});
  std::cout << table->ToString();

  // Choose compression
  std::shared_ptr<parquet::WriterProperties> props =
      parquet::WriterProperties::Builder().compression(arrow::Compression::SNAPPY)->build();

  // Opt to store Arrow schema for easier reads back into Arrow
  std::shared_ptr<parquet::ArrowWriterProperties> arrow_props =
      parquet::ArrowWriterProperties::Builder().store_schema()->build();

  std::shared_ptr<arrow::io::FileOutputStream> outfile = arrow::io::FileOutputStream::Open("cpp_out.parquet").ValueOrDie();
  status = parquet::arrow::WriteTable(*table.get(), arrow::default_memory_pool(), outfile);
  if (!status.ok()) {
    std::cerr << status.message() << std::endl;
    return -1;
  }
  return 0;
}
