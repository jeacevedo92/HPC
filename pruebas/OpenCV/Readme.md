# How to install opencv

[Step by step](http://www.linuxhispano.net/2012/11/05/instalar-opencv-2-4-2-ubuntu-12-04/)

For possible install error:

      sudo apt-get install qt5-default


## Generate the executable

This part is easy, just proceed as with any other project using CMake:

      cd <displayImage_directory>
    cmake .
    make

##  Result

By now you should have an executable (called displayImage in this case). You just have to run it giving an image location as an argument, i.e.:

      ./displayImage dream.jpg
