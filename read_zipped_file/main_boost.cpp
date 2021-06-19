// main_boost.cpp
// g++ main_boost.cpp -o main_boost -lboost_iostreams
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/lzma.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <magic.h>
#include <iostream>
#include <fstream>
#include <string>

// from https://gist.github.com/vivithemage/9489378
std::string check_file_type(const std::string& filename) {
  const char *magic_full;
  magic_t magic_cookie = magic_open(MAGIC_MIME_TYPE);
  if (magic_cookie == NULL) {
    std::cerr << "Unable to initialize magic library.\n";
  }
  if (magic_load(magic_cookie, NULL) != 0) {
    std::cerr << "Cannot load magic database "
              << magic_error(magic_cookie) << std::endl;
    magic_close(magic_cookie);
  }
  magic_full = magic_file(magic_cookie, filename.c_str());
  std::string result(magic_full);
  magic_close(magic_cookie);
  return result;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "No file specified.\n";
    return 1;
  }
  const std::string filename(argv[1]);
  const std::string mime_string = check_file_type(filename);
  // std::cout << mime_string << std::endl;
  std::ifstream ifs_compressd_in(filename.c_str());
  if (!ifs_compressd_in.is_open()) {
    std::cerr << "Error on opening file " << filename << std::endl;
    return 1;
  }
  boost::iostreams::filtering_istream in;
  if (mime_string.find("xz") != std::string::npos) {
    in.push(boost::iostreams::lzma_decompressor());
  } else if (mime_string.find("gzip") != std::string::npos) {
    in.push(boost::iostreams::gzip_decompressor());
  } else if (mime_string.find("bzip2") != std::string::npos) {
    in.push(boost::iostreams::bzip2_decompressor());
  }
  in.push(ifs_compressd_in);
  std::string line;
  while (std::getline(in, line)) {
    std::cout << line << std::endl;
  }
  return 0;
}
