
CXX=g++-8
FLAGS=-std=c++11
LIBS= -lmlpack -larmadillo -lboost_wserialization -lboost_atomic -lboost_chrono -lboost_container -lboost_context -lboost_contract -lboost_coroutine -lboost_date_time -lboost_exception -lboost_fiber -lboost_iostreams -lboost_json -lboost_locale -lboost_filesystem -lboost_log -lboost_log_setup -lboost_graph -lboost_json -lboost_locale -lboost_math_c99f -lboost_math_c99l -lboost_math_c99 -lboost_math_tr1 -lboost_math_tr1f -lboost_math_tr1l -lboost_stacktrace_noop -lboost_system -lboost_regex -lboost_random -lboost_python39 -lboost_prg_exec_monitor -lboost_nowide -lboost_program_options -lboost_stacktrace_addr2line -lboost_stacktrace_backtrace -lboost_stacktrace_basic -lboost_test_exec_monitor -lboost_thread -lboost_timer -lboost_type_erasure -lboost_unit_test_framework -lboost_wave -lboost_serialization

all: model controller

model: model.cpp model.h
	$(CXX) $(FLAGS) model.cpp -o model -DARMA_DONT_USE_WRAPPER -DARMA_USE_HDF5 -DDEBUG -lmlpack -lhdf5 -fopenmp

controller: controller.cpp controller.h
	$(CXX) $(FLAGS) controller.cpp -o controller $(LIBS) -fopenmp -DDEBUG
