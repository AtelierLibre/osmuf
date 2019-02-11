# OSMuf - OpenStreetMap for Urban Form
OSMuf is a project to explore the use of Python and OpenStreetMap for quantifying urban form, specifically the size and shape of urban/city blocks and built density.

It is in a large part inspired by the work of Geoff Boeing on OSMnx, Berghauser Pont and Haupt on the 'Spacemate', Barthélemy and Louf on Measuring Urban Form, and Philip Steadman on 'integrating 'Spacemate' with the work of Martin and March'.

Large scale studies of urban form often approximate city blocks by generating polygons from the centrelines of the transport network. The approach of OSMuf is to instead work with urban blocks as objects in themselves with their own spatial definition. Loosely this could be categorised as the difference between 'gross urban blocks' (defined by transport network centrelines) and 'net urban blocks' defined by the line separating privately owned land or land dedicated to public amenity from land dedicated to the transport network. The distinction is relevant because when urban form is studied at a smaller scale the relationship between, for example, individual street width and building height becomes important.

Urban Form and Density are recognised as having an impact on quality of life in urban environments. In terms of urban form, towers in otherwise empty estates are blamed for destroying the streetlife associated with traditional perimeter blocks. In terms of density, excessive residential densities are often blamed for poor quality of life and access to green space yet low residential densities can make provision of public services and amenities unfeasibly expensive.

This project aims to establish a method for quantifying existing built environments based on data from, or added to, OpenStreetMap and using an open source Python library based primarily on osmnx, geopandas and matplotlib.
