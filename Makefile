all:
	svn checkout http://asmlib-opencv.googlecode.com/svn/trunk/ lib/asmlib-opencv
	python lib/setup.py build

install:
	python lib/setup.py install