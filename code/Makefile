
CXX= g++
FLAGS= -std=c++11
LIBS= -DDEBUG -larmadillo -lmlpack -fopenmp

all: model controller

model: model.cpp model.h
	$(CXX) $(FLAGS) model.cpp -o model $(LIBS)

controller: controller.cpp controller.h
	$(CXX) $(FLAGS) controller.cpp -o controller $(LIBS)
