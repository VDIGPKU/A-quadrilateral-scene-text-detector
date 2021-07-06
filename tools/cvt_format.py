import os.path as osp
import os
import sys
import math
def validate_clockwise_points(points):
    """
    Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
    """
    
    if len(points) != 8:
        raise Exception("Points list not valid." + str(len(points)))
    
    point = [
                [int(points[0]) , int(points[1])],
                [int(points[2]) , int(points[3])],
                [int(points[4]) , int(points[5])],
                [int(points[6]) , int(points[7])]
            ]
    edge = [
                ( point[1][0] - point[0][0])*( point[1][1] + point[0][1]),
                ( point[2][0] - point[1][0])*( point[2][1] + point[1][1]),
                ( point[3][0] - point[2][0])*( point[3][1] + point[2][1]),
                ( point[0][0] - point[3][0])*( point[0][1] + point[3][1])
    ]
    
    summatory = edge[0] + edge[1] + edge[2] + edge[3];
    return summatory <= 0
def get_cent(poly):
    x = sum(poly[::2]) / 4
    y = sum(poly[1::2]) / 4
    return x, y


def angle_sort(coordinates):
    cx, cy = get_cent(coordinates)
    pts = [(coordinates[i], coordinates[i + 1]) for i in range(0, 8, 2)]
    pts.sort(key=lambda a: math.atan2(a[1] - cy, a[0] - cx))
    coordinates = []
    for p in pts:
        coordinates.append(p[0])
        coordinates.append(p[1])
    return coordinates

# method = 'icdar_aug_poly'
method = sys.argv[1]
img_list = '/home/liuyudong/icdar-notb/data/icdar/test/img_list.txt'
input_base = '/home/liuyudong/icdar-notb/data/icdar/result/{}'.format(method)
out_base = 'submit/{}'.format(method)
if not osp.isdir(out_base):
    os.makedirs(out_base)

len_cnt = []
tot=0
for i in range(30):len_cnt.append(0)
names = open(img_list).read().strip().split('\n')
for name in names:
    fout = open(osp.join(out_base, 'res_img_' + name.split('_')[-1] + '.txt'), 'w')
    with open(osp.join(input_base, name + '.txt')) as fin:
        for line in fin:
            score = float(line.strip().split()[-1])
            if score < 0.92: continue
            box = list(map(int, line.strip().split()[:-1]))[1:]
            if len(box) == 0: continue
            x2,y2=max(box[0::2]),max(box[1::2])
            x1,y1=min(box[0::2]),min(box[1::2])
            min_len = min(x2-x1+1,y2-y1+1)
            tm=0
            while(min_len>16*tm):tm+=1
            len_cnt[tm-1]+=1
            #assert len(box)==8,'len is {}'.format(len(box))
            #box = [ box[0],box[1],box[2],box[1],box[2],box[3],box[0],box[3] ]
            if len(box) == 0: continue
            for i, b in enumerate(box):
                box[i] = max(0, b)
            #box = angle_sort(box)
            if not validate_clockwise_points(box):
                print(box)
                continue
            tot+=1
            fout.write('{}'.format(','.join(str(b) for b in box)))
            fout.write('\r\n')
#for i in range(20):
#    print('min_len between {} and {} is {}'.format(16*i,16*i+16,len_cnt[i]))
#print('tot is {}'.format(tot))
cmd = 'zip -j -q submit/{}.zip submit/{}/*.txt'.format(method, method)
os.system(cmd)

