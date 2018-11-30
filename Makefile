.PHONY: all
all: main graph.pb

main: main.cpp session_config.h
	g++ -o main -std=c++17 -pthread main.cpp -ltensorflow

graph.pb: make_graph.py
	./make_graph.py

session_config.h: make_session_config.py
	./make_session_config.py > $@
