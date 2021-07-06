import cv2
f = open('name_poly/res_img_1.txt','r')
im = cv2.imread('test/img/img_1.jpg')
for line in f:
  line = line.strip()
  line = line.split(',')
  line = list(map(int,line[0:8]))
  cv2.line( im, (line[0],line[1]),(line[2],line[3]) ,(255,0,0),5)
  cv2.line( im, (line[2],line[3]),(line[4],line[5]) ,(255,0,0),5)
  cv2.line( im, (line[4],line[5]),(line[6],line[7]) ,(255,0,0),5)
  cv2.line( im, (line[6],line[7]),(line[0],line[1]) ,(255,0,0),5)

cv2.imwrite('img_1.jpg',im)
