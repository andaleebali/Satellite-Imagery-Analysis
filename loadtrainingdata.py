from osgeo import gdal
from osgeo import osr

def readTiff(mapfile):
    '''
    Reads geotiff file 
    '''

    images = []
    labels = []
    
    with open(mapfile) as file:
        for line in file.readlines():
            element = line.strip('\n')
            element = element.replace('\\','/')
            element = element.split()
            images.append('/home/s1885898/scratch/data/labely/' + element[0])
            labels.append('/home/s1885898/scratch/data/labely/' + element[1])




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
    readTiff(mapfile='/home/s1885898/scratch/data/labely/map.txt')