"""A series of read functions adjusted from bvbabel (credit to Faruk Gulban).
Functions, in contrast to bvbabel, are designed to read partial binary files,
Userful for senaty checks and quick plotting functions of single slices / single volumes
functions tested on brainvoyaer version 22.2. 
for any help email jjg.vanharen@maastrichtuniversity.nl"""

# import things we need
import numpy as np
import bvbabel

## FUNCTIONS

def read_stc_single_volume(filename, what_volume, nr_slices, nr_volumes, res_x, res_y, data_type=2):
    """Read Brainvoyager STC file - ignore all but a single volume.
    Adjusted from bvbabel (credit to Faruk Gulban) to only load single volume 
    Parameters
    ----------
    filename : string or file
        Path to file or binary file.
    what_volume: integer
        What volume to read
    nr_slices: integer
        Number of slices in each measurement. Referred to as "NrOfSlices"
        within the FMR text file.
    nr_volumes: integer
        Number of measurements (also called volumes or TRs). Referred to as
        "NrOfVolumes" within the FMR text file.
    res_x: integer
        Number of voxels along each row in each slice. Referred to as
        "ResolutionX" within the FMR text file.
    res_y: integer
        Number of voxels along each column in each slice. Referred to as
        "ResolutionY" within the FMR text file.
    data_type: integer, 1 or 2
        Each data element (intensity value) is represented either in 2 bytes
        (unsigned short) or in 4 bytes (float, default) as determined by the
        "DataType" entry in the FMR file.
    Returns
    -------
    data : 3D numpy.array, (x, y, slices)
        Image data.
    """
    
    with open(filename, 'rb') as f:
        
        # define full volume empty array
        if data_type == 1:
            data_img = np.zeros([res_x, res_y, nr_slices],
                                dtype = "<H")
        elif data_type == 2:
            data_img = np.zeros([res_x, res_y, nr_slices],
                                dtype = "<f")
        
        # loop over all slices
        for sl in range(nr_slices):
            
            # point towards correct bits in binary data
            if data_type == 1:
                f.seek( 2 * ((res_x * res_y * nr_volumes * sl) + (res_x * res_y * what_volume)) )
            elif data_type == 2:
                f.seek( 4 * ((res_x * res_y * nr_volumes * sl) + (res_x * res_y * what_volume)) )
    
            # load single slice and fill in
            data_img[:,:,sl] = read_stc_single_slice(f, 0, 0,
                                                     nr_slices=nr_slices,
                                                     nr_volumes=nr_volumes,
                                                     res_x=res_x,
                                                     res_y=res_y,
                                                     data_type=data_type)
    
    return data_img

def read_stc_single_slice(filename, what_slice, what_volume, nr_slices, nr_volumes, res_x, res_y, data_type=2):
    """Read Brainvoyager STC file - ignore all but a single slice of a single volume.
    Adjusted from bvbabel (credit to Faruk Gulban) to only load single slice 
    Parameters
    ----------
    filename : string or file
        Path to file or binary file.
    what_slice: integer
        What slice to read
    what_volume: integer
        What volume to read
    nr_slices: integer
        Number of slices in each measurement. Referred to as "NrOfSlices"
        within the FMR text file.
    nr_volumes: integer
        Number of measurements (also called volumes or TRs). Referred to as
        "NrOfVolumes" within the FMR text file.
    res_x: integer
        Number of voxels along each row in each slice. Referred to as
        "ResolutionX" within the FMR text file.
    res_y: integer
        Number of voxels along each column in each slice. Referred to as
        "ResolutionY" within the FMR text file.
    data_type: integer, 1 or 2
        Each data element (intensity value) is represented either in 2 bytes
        (unsigned short) or in 4 bytes (float, default) as determined by the
        "DataType" entry in the FMR file.
    Returns
    -------
    data : 2D numpy.array, (x, y)
        Image data.
    """
    
    # calculate the loading length 
    countlen = res_x * res_y
    offsetlen = (countlen * nr_volumes * what_slice) + (countlen * what_volume)

    if data_type == 1:
        data_img = np.fromfile(filename, dtype="<H", count=countlen, sep="",
                               offset=offsetlen * 2)  # calulate offset in bits (H = 2)
    elif data_type == 2:
        data_img = np.fromfile(filename, dtype="<f", count=countlen, sep="",
                               offset=offsetlen * 4)  # calulate offset in bits (f = 4)

    data_img = np.reshape(data_img, (res_x, res_y))
    data_img = np.transpose(data_img, (1, 0))
    data_img = data_img[:,::-1]
    
    return data_img

def read_fmr_header(filename):
    """Read Brainvoyager FMR header file
    adjusted from bvbabel (credit to Faruk Gulban) to be 
    lighter and ignore stc file.
    Parameters
    ----------
    filename : string
        Path to file.
    Returns
    -------
    header : dictionary
        Pre-data and post-data headers.
    """
    header = dict()
    info_pos = dict()
    info_tra = dict()
    info_multiband = dict()

    with open(filename, 'r') as f:
        lines = f.readlines()
        for j in range(0, len(lines)):
            line = lines[j]
            content = line.strip()
            content = content.split(":", 1)
            content = [i.strip() for i in content]

            # -----------------------------------------------------------------
            # NOTE[Faruk]: Quickly skip entries starting with number. This is
            # because such entries belong to other structures and are dealth
            # with below in transformations and multiband sections
            if content[0].isdigit():
                pass
            elif content[0] == "FileVersion":
                header[content[0]] = content[1]
            elif content[0] == "NrOfVolumes":
                header[content[0]] = int(content[1])
            elif content[0] == "NrOfSlices":
                header[content[0]] = int(content[1])
            elif content[0] == "NrOfSkippedVolumes":
                header[content[0]] = content[1]
            elif content[0] == "Prefix":
                header[content[0]] = content[1].strip("\"")
            elif content[0] == "DataStorageFormat":
                header[content[0]] = int(content[1])
            elif content[0] == "DataType":
                header[content[0]] = int(content[1])
            elif content[0] == "TR":
                header[content[0]] = content[1]
            elif content[0] == "InterSliceTime":
                header[content[0]] = content[1]
            elif content[0] == "TimeResolutionVerified":
                header[content[0]] = content[1]
            elif content[0] == "TE":
                header[content[0]] = content[1]
            elif content[0] == "SliceAcquisitionOrder":
                header[content[0]] = content[1]
            elif content[0] == "SliceAcquisitionOrderVerified":
                header[content[0]] = content[1]
            elif content[0] == "ResolutionX":
                header[content[0]] = int(content[1])
            elif content[0] == "ResolutionY":
                header[content[0]] = int(content[1])
            elif content[0] == "LoadAMRFile":
                header[content[0]] = content[1].strip("\"")
            elif content[0] == "ShowAMRFile":
                header[content[0]] = content[1]
            elif content[0] == "ImageIndex":
                header[content[0]] = content[1]
            elif content[0] == "LayoutNColumns":
                header[content[0]] = content[1]
            elif content[0] == "LayoutNRows":
                header[content[0]] = content[1]
            elif content[0] == "LayoutZoomLevel":
                header[content[0]] = content[1]
            elif content[0] == "SegmentSize":
                header[content[0]] = content[1]
            elif content[0] == "SegmentOffset":
                header[content[0]] = content[1]
            elif content[0] == "NrOfLinkedProtocols":
                header[content[0]] = content[1]
            elif content[0] == "ProtocolFile":
                header[content[0]] = content[1].strip("\"")
            elif content[0] == "InplaneResolutionX":
                header[content[0]] = content[1]
            elif content[0] == "InplaneResolutionY":
                header[content[0]] = content[1]
            elif content[0] == "SliceThickness":
                header[content[0]] = content[1]
                # NOTE[Faruk]: This is duplicate entry that appears in position
                # information header too. I decided to use the last occurance
                # as the true value for both header entries. These two entries
                # should always match if the source file is not manipulated in
                # some way.
                info_pos[content[0]] = content[1]
            elif content[0] == "SliceGap":
                header[content[0]] = content[1]
            elif content[0] == "VoxelResolutionVerified":
                header[content[0]] = content[1]

            # -----------------------------------------------------------------
            # Position information
            elif content[0] == "PositionInformationFromImageHeaders":
                pass  # No info to be stored here
            elif content[0] == "PosInfosVerified":
                info_pos[content[0]] = content[1]
            elif content[0] == "CoordinateSystem":
                info_pos[content[0]] = content[1]
            elif content[0] == "Slice1CenterX":
                info_pos[content[0]] = content[1]
            elif content[0] == "Slice1CenterY":
                info_pos[content[0]] = content[1]
            elif content[0] == "Slice1CenterZ":
                info_pos[content[0]] = content[1]
            elif content[0] == "SliceNCenterX":
                info_pos[content[0]] = content[1]
            elif content[0] == "SliceNCenterY":
                info_pos[content[0]] = content[1]
            elif content[0] == "SliceNCenterZ":
                info_pos[content[0]] = content[1]
            elif content[0] == "RowDirX":
                info_pos[content[0]] = content[1]
            elif content[0] == "RowDirY":
                info_pos[content[0]] = content[1]
            elif content[0] == "RowDirZ":
                info_pos[content[0]] = content[1]
            elif content[0] == "ColDirX":
                info_pos[content[0]] = content[1]
            elif content[0] == "ColDirY":
                info_pos[content[0]] = content[1]
            elif content[0] == "ColDirZ":
                info_pos[content[0]] = content[1]
            elif content[0] == "NRows":
                info_pos[content[0]] = content[1]
            elif content[0] == "NCols":
                info_pos[content[0]] = content[1]
            elif content[0] == "FoVRows":
                info_pos[content[0]] = content[1]
            elif content[0] == "FoVCols":
                info_pos[content[0]] = content[1]
            elif content[0] == "SliceThickness":
                # NOTE[Faruk]: This is duplicate entry that appears twice.
                # ee header['SliceThickness'] section above.
                pass
            elif content[0] == "GapThickness":
                info_pos[content[0]] = content[1]

            # -----------------------------------------------------------------
            # Transformations section
            elif content[0] == "NrOfPastSpatialTransformations":
                info_tra[content[0]] = int(content[1])
            elif content[0] == "NameOfSpatialTransformation":
                info_tra[content[0]] = content[1]
            elif content[0] == "TypeOfSpatialTransformation":
                info_tra[content[0]] = content[1]
            elif content[0] == "AppliedToFileName":
                info_tra[content[0]] = content[1]
            elif content[0] == "NrOfTransformationValues":
                info_tra[content[0]] = content[1]

                # NOTE(Faruk): I dont like this matrix reader but I don't see a
                # more elegant way for now.
                nr_values = int(content[1])
                affine = []
                v = 0  # Counter for values
                n = 1  # Counter for lines
                while v < nr_values:
                    line = lines[j + n]
                    content = line.strip()
                    content = content.split()
                    for val in content:
                        affine.append(float(val))
                    v += len(content)  # Count values
                    n += 1  # Iterate line
                affine = np.reshape(np.asarray(affine), (4, 4))
                info_tra["Transformation matrix"] = affine

            # -----------------------------------------------------------------
            # This part only contains a single information
            elif content[0] == "LeftRightConvention":
                header[content[0]] = content[1]

            # -----------------------------------------------------------------
            # Multiband section
            elif content[0] == "FirstDataSourceFile":
                info_multiband[content[0]] = content[1]
            elif content[0] == "MultibandSequence":
                info_multiband[content[0]] = content[1]
            elif content[0] == "MultibandFactor":
                info_multiband[content[0]] = content[1]
            elif content[0] == "SliceTimingTableSize":
                info_multiband[content[0]] = int(content[1])

                # NOTE(Faruk): I dont like this matrix reader but I don't see a
                # more elegant way for now.
                nr_values = int(content[1])
                slice_timings = []
                for n in range(1, nr_values+1):
                    line = lines[j + n]
                    content = line.strip()
                    slice_timings.append(float(content))
                info_multiband["Slice timings"] = slice_timings

            elif content[0] == "AcqusitionTime":
                info_multiband[content[0]] = content[1]

    header["Position information"] = info_pos
    header["Transformation information"] = info_tra
    header["Multiband information"] = info_multiband

    return header