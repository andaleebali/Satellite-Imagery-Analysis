from osgeo import gdal
from osgeo import osr

def readTiff(filename):
    '''
    Reads geotiff file 
    '''
    file = gdal.Open(filename)

    nX=file.RasterXSize
    nY=file.RasterYSize

    transform_file=file.GetGeoTransform()
    xOrigin=transform_file[0]
    yOrigin=transform_file[3]
    pixelWidth=transform_file[1]
    pixelHeight=transform_file[5]

    red=file.GetRasterBand(1).ReadAsArray(0,0,nX,nY)
    green=file.GetRasterBand(2).ReadAsArray(0,0,nX,nY)
    blue=file.GetRasterBand(3).ReadAsArray(0,0,nX,nY)
    
    nir=file.GetRasterBand(4).ReadAsArray(0,0,nX,nY)



if __name__=="__main__":
    readTiff(filename='/home/s1885898/scratch/data/Subset3.tif')