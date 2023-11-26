# Define the compiler
CXX = g++

# Define compiler flags
CXXFLAGS = -Wall -Wextra -std=c++17

# Define the name of the executable file
TARGET = raytracer

# Define the source files
SRCS = raytracer.cpp

# Define the object files
OBJS = $(SRCS:.cpp=.o)

# Define the build target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Clean up
clean:
	$(RM) $(TARGET) $(OBJS)

# Phony targets
.PHONY: all clean
