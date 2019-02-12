# OSMuf - OpenStreetMap for Urban Form
OSMuf is a project to explore the use of Python and OpenStreetMap for quantifying urban form, specifically the size and shape of urban/city blocks and built density.

It is in a large part inspired by the work of Geoff Boeing on OSMnx, Berghauser Pont and Haupt on the 'Spacemate', Barthélemy and Louf on Measuring Urban Form, and Philip Steadman on 'integrating 'Spacemate' with the work of Martin and March'.

### Urban blocks - net or gross?

Large scale studies of urban form often approximate city blocks by generating polygons from the centrelines of the transport network. The approach of OSMuf is to instead work with urban blocks as objects in themselves with their own spatial definition. Loosely this could be categorised as the difference between 'gross urban blocks' defined by transport network centrelines and 'net urban blocks' defined by the line separating privately owned land or land dedicated to public amenity from land dedicated to the transport network.

### Urban blocks as part of the urban fabric

There are benefits to dealing with urban blocks directly rather than just as the resultant geometry from the transport network. First, as others have noted, when polygonising the transport network cul-de-sacs tend to disappear - this is misleading especially when dealing with perimeter blocks because length of accessible frontage has a much greater impact on developable area than the absolute area of the urban block. Second, centrelines on their own say little about street width and, as many studies have emphasised, the resultant angle from the street width and building height is often used as a proxy for available daylight within buildings. Third, the limits of urban blocks are generally observable in the real world - this is important for being able to measure density against data available freely in OpenStreetMap which is a topographic map not a cadastre.

### Urban form, density and quality of life

Urban Form and Density are recognised as having an impact on quality of life in urban environments. In terms of urban form, towers in otherwise empty estates are blamed for destroying the streetlife associated with traditional perimeter blocks. In terms of density, excessive residential densities are blamed for poor quality of life with limited access to green space yet low residential densities can make provision of public services and amenities unfeasibly expensive.

### An evidence base

This project aims to establish a method for quantifying existing built environments based on data from, or added to, OpenStreetMap and using an open source Python library based primarily on osmnx, geopandas and matplotlib. Ultimately it aims to put some relatively simple measures against a variety of urban environments to contribute to an informed discussion about how best to design sustainable towns and cities.
