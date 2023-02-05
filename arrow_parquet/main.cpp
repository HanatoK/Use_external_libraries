#include <arrow/api.h>
#include <arrow/chunked_array.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/type_fwd.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <memory>
#include <iostream>

// Basic data structures
arrow::Status RunBasicArrowOperations() {
  // construct an array using ArrayBuilder
  arrow::Int64Builder builder;
  ARROW_RETURN_NOT_OK(builder.Append(1));
  ARROW_RETURN_NOT_OK(builder.Append(2));
  ARROW_RETURN_NOT_OK(builder.Append(3));
  ARROW_RETURN_NOT_OK(builder.AppendNull());
  ARROW_RETURN_NOT_OK(builder.Append(5));
  ARROW_RETURN_NOT_OK(builder.Append(6));
  ARROW_RETURN_NOT_OK(builder.Append(7));
  ARROW_RETURN_NOT_OK(builder.Append(8));
  auto maybe_array = builder.Finish();
  if (!maybe_array.ok()) {
    return maybe_array.status();
  }
  // shared ptr to the array
  auto array = maybe_array.ValueOrDie();
  // setup fields
  auto field_col_a = arrow::field("value_a", arrow::int64());
  // setup schema (collumn description)
  auto schema = arrow::schema({field_col_a});
  // make a RecordBatch (continuous table)
  auto rbatch = arrow::RecordBatch::Make(schema, array->length(), {array});
  // print the array
  std::cout << rbatch->ToString();
  // chunked array
  arrow::ArrayVector val_vecs{array, array};
  auto chunks = std::make_shared<arrow::ChunkedArray>(val_vecs);
  // make a table
  auto table = arrow::Table::Make(schema, {chunks}, chunks->length());
  std::cout << "Table:\n" << table->ToString();
  return arrow::Status::OK();
}

int main() {
  arrow::Status st = RunBasicArrowOperations();
  if (!st.ok()) {
    std::cerr << st << std::endl;
    return 1;
  }
  return 0;
}
