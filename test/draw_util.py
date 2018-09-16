import numpy as np

def output_points(filename,pts,colors=None):
    has_color=pts.shape[1]>=6
    with open(filename, 'w') as f:
        for i,pt in enumerate(pts):
            if colors is None:
                if has_color:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],int(pt[3]),int(pt[4]),int(pt[5])))
                else:
                    f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))

            else:
                if len(colors.shape)==2:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],colors[i,0],colors[i,1],colors[i,2]))
                else:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],colors[0],colors[1],colors[2]))