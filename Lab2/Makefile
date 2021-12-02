
CC = g++
CFLAGS = -Wall -O0 

SRC = loadImage.cpp
OBJS = 

LIBS = -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
CFLAGS += -I /usr/include/opencv2

all: loadImage

loadImage: loadImage.cpp $(OBJS)
	$(CC) -o loadImage loadImage.cpp $(OBJS) $(CFLAGS) $(LIBS)

clean:
	rm -f *.o loadImage