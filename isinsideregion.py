# define function that verifies the distance
def isinsideregion(x_, y_, r):
    if (r[0] <= x_ <= r[2] or r[2] <= x_ <= r[0]) and (r[3] <= y_ <= r[1] or r[1] <= y_ <= r[3]):
        return True
    else:
        return False


# Para o retangulo:
# x1<x<x2
# y>y2 y<y1   y2 <= y <= y1