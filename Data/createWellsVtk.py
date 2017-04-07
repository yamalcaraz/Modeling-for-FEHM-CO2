#-------------------------------------------------------------------------------
# Name:        
# Purpose:     1. Converting actual coordinates to model coordinates

# Author:      alcaraz.jt
#
# Update:     2/5/2016
#
#-------------------------------------------------------------------------------

from vtk import vtkUnstructuredGrid, vtkDoubleArray, vtkUnstructuredGridWriter, vtkXMLUnstructuredGridWriter, vtkStringArray, \
    vtkPoints, vtkCellArray, vtkPolyLine, VTK_POLY_LINE, vtkXMLRectilinearGridReader, vtkRectilinearGridGeometryFilter
from numpy import cos, sin, sqrt
from math import radians
import pandas as pd
from pdb import set_trace
from collections import Counter
import numpy as np

def sampleUsage():
    data_dir = "D:/Users/Yam/Desktop/Thesis/Modeling Directory/Data/"
    wells_file = data_dir + "MGPF Deviation Survey and Well Data1.xlsx"
    well_list = data_dir + "ModelWellList.xlsx"
    
    wells = pd.read_excel(well_list)
    wells = wells.values.flatten().astype(str)
    wells = wells[wells!='nan']
    
    wells_DS=pd.read_excel(wells_file,sheetname=None)
    
    welltracks_df=pd.DataFrame(columns=['Well', 'X', 'Y', 'Z'])
    for w in wells_DS.keys():
        if w in wells:
            temp=wells_DS[w].loc[:,['Easting','Northing','MRSL']]
            temp.columns=['X','Y','Z']
            temp['Well']=w
            welltracks_df=welltracks_df.append(temp)
    welltracks_df=welltracks_df.reset_index().sort_values(['Well','Z'])
    #create vtk for wells
#    new_origin = (521460.0927, 1003382.334)
#    deg_rotation = 51.0
#    welltracks_df = convertDeviationSurvey('DS_v4.1_4Feb2016.xlsx',new_origin,deg_rotation)
    makeVTKWells('Wells_MGPF', welltracks_df, xml = True)
    
    #find well blocks in paraview
    query_string = findWellBlocks('BL1D', welltracks_df, 'T2-R22_results_1.vtr')
    print query_string
    
class Point:
    def __init__(self, (x, y, z)):
        self.x = x
        self.y = y
        self.z = z
        self.coords = (x, y, z)

class Segment:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.length = sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2 + (p2.z-p1.z)**2)

    def divide(self, num_segments):
        x_points = np.linspace(self.p1.x, self.p2.x, num_segments)
        y_points = np.linspace(self.p1.y, self.p2.y, num_segments)
        z_points = np.linspace(self.p1.z, self.p2.z, num_segments)
        line_points = [Point(i) for i in zip(x_points,y_points,z_points)]
        return line_points 
        
def transformCoords(x, y, origin_coords, deg_rotation):
    rotation_rad = radians(deg_rotation) #degrees of rotation from actual to model
    #translation
    x_translated = x - origin_coords[0]
    y_translated = y - origin_coords[1]
    #rotation
    x_rotated = x_translated*cos(rotation_rad) + y_translated*sin(rotation_rad)
    y_rotated = - x_translated*sin(rotation_rad) + y_translated*cos(rotation_rad)
    return (x_rotated,y_rotated)
         
def convertDeviationSurvey(fname, origin_coords, deg_rotation):
    welltracks = pd.read_excel(fname)
    x, y = transformCoords(welltracks['Easting'], welltracks['Northing'],origin_coords, deg_rotation)
    welltracks['X'] = x
    welltracks['Y'] = y
    return welltracks
    
def makeVTKWells(fname_base, welltracks_df, xml=False):
    """Creates a vtk Unstructured Grid file (*.vtk, *.vtu) from a welltracks DataFrame 
    
    Parameters:
        fname_base -- the output filename will be [fname_base].vtk or [fname_base].vtu for xml format
        welltracks_df -- DataFrame contaning 'X', 'Y', and 'Elev_mASL' columns.
                         This is created using the transformCoords function.
        xml -- set to True if xml format is preferred
    """ 
    numpoints = welltracks_df.shape[0]
    wells = welltracks_df['Well'].unique().tolist()
    numwells = len(wells)

    grid = vtkUnstructuredGrid()
    points = vtkPoints()  
    
    for i in range(numpoints):
        points.InsertNextPoint(welltracks_df.loc[i,'X'], welltracks_df.loc[i,'Y'], welltracks_df.loc[i,'Z'])
    
    cells = vtkCellArray()
    wellname = vtkStringArray()
    wellname.SetName('Well')
    
    for well in wells:
        print well
        polyline = vtkPolyLine()
        indices = welltracks_df[welltracks_df['Well']==well].index.tolist()
        for i, j in enumerate(indices):
            polyline.GetPointIds().SetNumberOfIds(len(indices))
            polyline.GetPointIds().SetId(i,j)
            
        cells.InsertNextCell(polyline)
        wellname.InsertNextValue(well)
        
    grid.SetPoints(points)
    grid.SetCells(VTK_POLY_LINE, cells)
    grid.GetCellData().AddArray(wellname)
    
    if xml:
        writer = vtkXMLUnstructuredGridWriter()
        writer.SetFileName('{}.vtu'.format(fname_base))
        writer.SetDataModeToAscii()
        writer.SetInputData(grid)
        writer.Write()
        
    else:
        writer = vtkUnstructuredGridWriter()
        writer.SetFileName('{}.vtk'.format(fname_base))
        writer.SetInputData(grid)
        writer.Write()

def pointInRecPrism(point, rec_prism_points):
    return all([rec_prism_points[0][i]<=point[i]<=rec_prism_points[1][i] for i in xrange(3)])
    
def readVTK(fname):
    reader = vtkXMLRectilinearGridReader()
    reader.SetFileName(fname)
    reader.Update()
    grid = reader.GetOutput(0)
    return grid

def getCellFromPoint(point, grid):
    cell_found=False
    for i_cell in xrange(grid.GetNumberOfCells()):
        cell =  grid.GetCell(i_cell)
        points = cell.GetPoints()
        rec_prism_points = [points.GetPoint(4),points.GetPoint(3)] #opposite corners of the prism
        if pointInRecPrism(point, rec_prism_points):
            cell_found = True
            break
        
    if not cell_found:
        print "Point {} not found in model.".format(point)
        return None, None
    else:
        return i_cell, cell

def getArrayFromCell(cell_ids, grid, array_name):
    data = []
    if not isinstance(cell_ids, list):
        cell_ids = [cell_ids]
    for i in cell_ids:
        data += [grid.GetCellData().GetScalars(array_name).GetValue(i)]
    return data
    
def findIntersectedBlocks(well_name, welltracks_df, grid):
    well_df = welltracks_df[welltracks_df['Well']==well_name]
    points = zip(well_df['X'],well_df['Y'],well_df['Elev_mASL'])
    intersected_cells = []
    for p in points:
        id_cell, cell = getCellFromPoint(p, grid)
        if cell:
            intersected_cells += [id_cell]
    return np.unique(intersected_cells)

def createQueryString(query_dict):
    """Creates a query string from a dictionary
    
    Parameters:
        query_dict -- format is {array_name1:[value1, value2, ...], array_name2:[value1, value2, ...]}
    """ 
    query_string = ''
    (id == 1) | (id == 2) | (id == 3) | (id == 4)
    for k, l in query_dict.iteritems():
        for v in l:
            query_string += '({0}=={1})|'.format(k,v)
    query_string = query_string[:-1]
    return query_string

def findWellBlocks(well_name, welltracks_df, vtr_file):
    """Find the Cell IDs of a well in Paraview. Creates a query string that can be copied to Paraview's find function.
    
    Parameters:
        well_name -- name of well
        welltracks_df -- DataFrame contaning 'X', 'Y', and 'Elev_mASL' columns.
                         This is created using the transformCoords function.
        vtr_file -- a vtr file containing the grid
    """ 
    grid = readVTK(vtr_file)
    ids = findIntersectedBlocks(well_name, welltracks_df, grid)
    query_dict = {'id':ids}
    return createQueryString(query_dict)

#if __name__ == '__main__':
#    main()