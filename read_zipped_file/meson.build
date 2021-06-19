project('read_zipped_file', 'cpp')
boost_iostreams_dep = dependency('boost', modules : ['iostreams'])
file_dep = dependency('libmagic')
src = ['main_boost.cpp']
executable('read_zipped_file', src, cpp_args : ['-std=c++17'], dependencies: [boost_iostreams_dep, file_dep])