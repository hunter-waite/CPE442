CC = g++
CFLAGS = -Wall -O0 

SRC = sobel.cpp
OBJS = 

LIBS = -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lpthread -mfpu=neon
CFLAGS += -I /usr/include/opencv2

all: sobel

sobel: sobel.cpp $(OBJS)
	$(CC) -o sobel sobel.cpp $(OBJS) $(CFLAGS) $(LIBS)

clean:
	rm -f *.o sobel