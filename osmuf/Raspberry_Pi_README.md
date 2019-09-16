# Installation on RaspberryPi

*updated 15 September 2019*

OSMuf v0.1 has been successfully installed and run on a RaspberryPi 3 Model B.

It was installed on 'Raspbian Stretch with Desktop' (v9.11) and the included Python 3.5.3.

The steps below are provided for information only. They are not the minimum number of steps - some of them are redundant - but they did result in a working installation. Development of OSMuf is not targeted at the Raspberry Pi but, as a potential tool for communities interested in measuring building density at a neighbourhood scale, demonstrating that it can be run on low-cost hardware is interesting.

These steps may be refined in the future. At the moment they are provided 'as is'. Sometimes grouping the install of the packages seems to make apt fail and splitting the installation commands seemed to be more robust. Occasionally the `pip3 install` steps also generated errors but these were resolved by simply repeating the command.

```
$ sudo apt install build-essential
$ sudo apt install python3-dev
$ sudo apt install python3-setuptools
$ sudo apt install python3-pip
$ sudo apt install python3-wheel
$ sudo apt install python3-scipy python3-distlib libzmq-dev
$ sudo apt install libgdal-dev
$ sudo apt install python3-matplotlib
$ sudo apt install xsel xclip libxml2-dev
$ sudo apt install libxslt-dev
$ sudo apt install python3-lxml python3-h5py python3-numexpr
$ sudo apt install python3-dateutil python3-six python3-tz python3-bs4
$ sudo apt install python3-html5lib python3-openpyxl python3-tables
$ sudo apt install python3-xlrd cython python3-sqlalchemy
$ sudo apt install python3-xlsxwriter python3-jinja2 python3-boto
$ sudo apt install python3-gflags python3-googleapi python3-httplib2
$ sudo apt install python3-zmq libspatialindex-dev

$ sudo apt update
$ sudo shutdown -r now

$ sudo pip3 install xlwt
$ sudo pip3 install bottleneck
$ sudo pip3 install rtree
$ sudo apt install python3-numpy python3-matplotlib python3-mpltoolkits.basemap python3-scipy python3-sklearn python3-pandas
$ sudo pip3 install statsmodels
$ sudo apt install python3-requests python3-pil python3-geopy python3-shapely python3-pyproj
$ sudo pip3 install scrapy
$ sudo pip3 install jupyter
$ sudo pip3 install geopandas
$ sudo pip3 install osmnx
$ sudo apt install libatlas-base-dev

$ jupyter notebook
```